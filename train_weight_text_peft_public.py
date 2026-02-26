import argparse
import gc
import itertools
import logging
import math
import os
import shutil
import warnings
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from packaging import version
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.models.lora import LoRALinearLayer
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, PeftModel
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0
from diffusers.loaders import AttnProcsLayers

def infer_race_from_caption(caption: str) -> str:
    c = caption.lower()

    # 1) Prefer explicit tags (most reliable)
    # Match whole-word-ish tokens so "race_mena," or "RACE_MENA" works.
    # if re.search(r"\brace_mena\b", c):
    #     return "mena"
    # if re.search(r"\brace_asian\b", c):
    #     return "asian"
    # if re.search(r"\brace_white\b", c):
    #     return "white"
    # if re.search(r"\brace_black\b", c):
    #     return "black"
    # if re.search(r"\brace_hispanic\b", c):
    #     return "hispanic"

    # 2) Fallback: your human-readable categories
    # Order matters: MENA before african/black, etc.
    if ("middle eastern" in c) or ("north african" in c) or ("mena" in c):
        return "mena"

    # "Black or African American"
    if ("black" in c) or ("african american" in c):
        return "black"

    if "asian" in c:
        return "asian"

    if ("white" in c) or ("caucasian" in c):
        return "white"

    # Hispanic (ethnicity)
    if ("hispanic" in c) or ("latino" in c) or ("latina" in c):
        return "hispanic"

    return "unknown"


# ----------------------------
# Two-phase training helpers
# ----------------------------
def freeze_module_params_(module):
    """Freeze all parameters in module."""
    for p in module.parameters():
        p.requires_grad_(False)

def set_param_group_lr_(optimizer, group_index: int, new_lr: float):
    """Safely set LR of an existing param group."""
    if group_index >= len(optimizer.param_groups):
        raise IndexError(f"param group {group_index} out of range (have {len(optimizer.param_groups)})")
    optimizer.param_groups[group_index]["lr"] = float(new_lr)

def any_trainable(module) -> bool:
    return any(p.requires_grad for p in module.parameters())


def inject_unet_lora_peft(unet, rank, dropout=0.0):
    cfg = LoraConfig(
        r=rank,
        lora_alpha=rank,
        lora_dropout=dropout,
        bias="none",
        # UNet attention projections
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    return get_peft_model(unet, cfg)


logger = get_logger(__name__)


def import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with prompts for fine-tuning the model.
    Supports per-image prompts via <image>.txt sidecar files.
    """

    def __init__(
            self,
            instance_data_root,
            instance_prompt,
            class_prompt,
            class_data_root=None,
            class_num=None,
            size=1024,
            repeats=1,
            center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop

        self.instance_prompt = instance_prompt          # fallback prompt
        self.class_prompt = class_prompt

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exist.")

        image_exts = {".png", ".jpg", ".jpeg", ".webp"}

        # ----- INSTANCE IMAGES + PER-IMAGE PROMPTS -----
        self.instance_images = []
        self.custom_instance_prompts = []  # now a real list

        instance_paths = sorted(self.instance_data_root.iterdir())

        for path in instance_paths:
            if path.suffix.lower() not in image_exts:
                continue

            img = Image.open(path)

            # Look for a per-image caption <filename>.txt
            caption_path = path.with_suffix(".txt")
            if caption_path.exists():
                with open(caption_path, "r") as f:
                    caption = f.read().strip()
                if not caption:
                    caption = instance_prompt  # fallback if empty
            else:
                caption = instance_prompt  # fallback if missing

            # Repeat image + caption
            for _ in range(repeats):
                self.instance_images.append(img)
                self.custom_instance_prompts.append(caption)

        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images

        # ----- CLASS DATA (PRIOR PRESERVATION) -----
        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())

            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)

            # Make total length the max (as in your original logic)
            self._length = max(self.num_class_images, self.num_instance_images)

        else:
            self.class_data_root = None

        # ----- TRANSFORMS (unchanged) -----
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
                # transforms.Normalize([0.5], [0.5]),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}

        # ----- INSTANCE BRANCH -----
        instance_image = self.instance_images[index % self.num_instance_images]
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        example["instance_images"] = self.image_transforms(instance_image)

        # Use per-image caption
        caption = self.custom_instance_prompts[index % self.num_instance_images]
        example["instance_prompt"] = caption

        # ----- CLASS BRANCH (optional) -----
        if self.class_data_root:
            class_image = Image.open(
                self.class_images_path[index % self.num_class_images]
            )
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")

            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt"] = self.class_prompt

        return example

    
def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        prompts += [example["class_prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {"pixel_values": pixel_values, "prompts": prompts}
    return batch

class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

def is_belong_to_groups(key: str, groups: list) -> bool:
    try:
        for g in groups:
            if key.startswith(g):
                return True
        return False
    except Exception as e:
        raise type(e)(f'failed to is_belong_to_groups, due to: {e}')

    
def main(
        pretrained_model_name_or_path,
        instance_data_dir,
        instance_prompt,
        output_dir,
        size=1024,
        repeats=1,
        center_crop=False,
        gradient_checkpointing=True,
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
        report_to="tensorboard",
        train_text_enoder=False,
        rank=64,
        seed=None,
        scale_lr=False,
        learning_rate=5e-5,
        lr_scheduler="constant",
        lr_num_cycles=1,
        lr_power=1.0,
        lr_warmup_steps=100,
        train_batch_size=1,
        max_train_steps=12000,
        num_train_epochs=1,
        optimizer="adamw",
        use_8bit_adam=True,
        adam_beta1=0.9,
        adam_beta2=0.999,
        # adam_weight_decay=1e-4,
        adam_weight_decay=0,
        adam_epsilon=1e-8,
        checkpointing_steps=1000
    ):
    logging_dir = Path(output_dir, "logs")
    _te_frozen = False  # state flag

    accelerator_project_config = ProjectConfiguration(
        project_dir=output_dir, logging_dir=logging_dir
    )

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs]
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if seed is not None:
        set_seed(seed)

    if accelerator.is_main_process:
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

    # Load Tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False
    )

    tokenizer_two = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=None,
        use_fast=False
    )

    # Import text encder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path,
        revision=None
    )

    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path,
        revision=None,
        subfolder="text_encoder_2"
    )

    # Load Scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        pretrained_model_name_or_path, subfolder="scheduler"
    )
    
    # Load Text Encoders
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=None
    )

    text_encoder_two = text_encoder_cls_two.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=None
    )

    # Freeze base weights (OK)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # Inject LoRA adapters using PEFT
    rank_text = 8  # e.g. 8 or 16
    te_lora_config = LoraConfig(
    r=rank_text,
    lora_alpha=rank_text,
    lora_dropout=0.0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )

    text_encoder_one = get_peft_model(text_encoder_one, te_lora_config)
    text_encoder_two = get_peft_model(text_encoder_two, te_lora_config)

    # Optional: print trainable counts
    print("Trainable TE1 params:", sum(p.numel() for p in text_encoder_one.parameters() if p.requires_grad))
    print("Trainable TE2 params:", sum(p.numel() for p in text_encoder_two.parameters() if p.requires_grad))

    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        revision=None
    )

    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="unet"
    )

    # Only train the adapter LoRA layers
    vae.requires_grad_(False)
    unet.requires_grad_(False) # freeze base
    unet = inject_unet_lora_peft(unet, rank=rank)
    unet.print_trainable_parameters()   # optional sanity check
    unet_lora_parameters = [p for p in unet.parameters() if p.requires_grad]

    # Set mixed precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move models to precision set
    unet.to(accelerator.device, dtype=weight_dtype)
    # text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    # text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=torch.float32)
    text_encoder_two.to(accelerator.device, dtype=torch.float32)

    # VAE is always at high precision
    vae.to(accelerator.device, dtype=torch.float32)

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if train_text_enoder:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()


    def debug_trainable_params(model, name="model"):
        total, trainable = 0, 0
        for n, p in model.named_parameters():
            num = p.numel()
            total += num
            if p.requires_grad:
                trainable += num
                print(f"TRAINABLE in {name}: {n} {tuple(p.shape)}")
        print(f"{name}: trainable {trainable} / total {total} parameters\n")

    debug_trainable_params(unet, "UNet")

    def save_model_hook(models, weights, output_dir):
        if not accelerator.is_main_process:
            return

        # unwrap references once
        unet_ref = accelerator.unwrap_model(unet)
        te1_ref  = accelerator.unwrap_model(text_encoder_one)
        te2_ref  = accelerator.unwrap_model(text_encoder_two)

        # Save each adapter into a subfolder of the checkpoint dir
        for model in models:
            m = accelerator.unwrap_model(model)

            if m is unet_ref:
                m.save_pretrained(os.path.join(output_dir, "unet_lora"))

            elif m is te1_ref:
                m.save_pretrained(os.path.join(output_dir, "te1_lora"))

            elif m is te2_ref:
                m.save_pretrained(os.path.join(output_dir, "te2_lora"))

            else:
                raise ValueError(f"unexpected save model: {m.__class__}")

            # prevent accelerate from saving full model weights
            weights.pop()


    def load_model_hook(models, input_dir):
        # unwrap references once
        unet_ref = accelerator.unwrap_model(unet)
        te1_ref  = accelerator.unwrap_model(text_encoder_one)
        te2_ref  = accelerator.unwrap_model(text_encoder_two)

        while len(models) > 0:
            model = models.pop()
            m = accelerator.unwrap_model(model)

            if m is unet_ref:
                adapter_dir = os.path.join(input_dir, "unet_lora")
                if os.path.isdir(adapter_dir):
                    model = PeftModel.from_pretrained(model, adapter_dir)

            elif m is te1_ref:
                adapter_dir = os.path.join(input_dir, "te1_lora")
                if os.path.isdir(adapter_dir):
                    model = PeftModel.from_pretrained(model, adapter_dir)

            elif m is te2_ref:
                adapter_dir = os.path.join(input_dir, "te2_lora")
                if os.path.isdir(adapter_dir):
                    model = PeftModel.from_pretrained(model, adapter_dir)

            else:
                raise ValueError(f"unexpected load model: {m.__class__}")


    accelerator.register_save_state_pre_hook(save_model_hook)
    # accelerator.register_load_state_pre_hook(load_model_hook)

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )
        
    # Optimization parameters
    te1_trainable = [p for p in text_encoder_one.parameters() if p.requires_grad]
    te2_trainable = [p for p in text_encoder_two.parameters() if p.requires_grad]

    params_to_optimize = [
        {"params": unet_lora_parameters, "lr": learning_rate},
        {"params": te1_trainable, "lr": learning_rate * 0.1},
        {"params": te2_trainable, "lr": learning_rate * 0.1},
    ]


    # Create Optimizer
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
            
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        params_to_optimize,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon
    )

    for i, g in enumerate(optimizer.param_groups):
        print(f"Group {i}: lr={g['lr']}  tensors={len(g['params'])}")

    # Create Dataset and DataLoader
    train_dataset = DreamBoothDataset(
        instance_data_root=instance_data_dir,
        instance_prompt=instance_prompt,
        class_prompt=None,
        class_data_root=None,
        class_num=None,
        size=size,
        repeats=repeats,
        center_crop=center_crop,
    )

    # ---- Build sampling weights to balance groups ----
    sample_groups = []
    num_instances = train_dataset.num_instance_images  # defined in your dataset
    group_counts = Counter()

    feat_re = re.compile(r"\bFEAT_[A-Z0-9_]+\b")
    feat_race_counts = defaultdict(Counter)  # FEAT_* -> Counter({race: n})

    for i in range(num_instances):
        caption = train_dataset.custom_instance_prompts[i].lower()

        race = infer_race_from_caption(caption)

        is_newborn = ("age_newborn" in caption) or ("newborn" in caption)
        is_older   = ("age_older" in caption) or ("child" in caption)
        is_synd    = ("cond_syndromic" in caption) or ("genetic syndrome" in caption)
        is_healthy = ("cond_healthy" in caption) or ("healthy" in caption)

        if is_newborn:
            if is_synd:
                g = f"newborn_syndromic_{race}"
            elif is_healthy:
                g = f"newborn_healthy_{race}"
            else:
                g = "unknown"
        elif is_older:
            if is_synd:
                g = f"older_syndromic_{race}"
            elif is_healthy:
                g = f"older_healthy_{race}"
            else:
                g = "unknown"
        else:
            g = "unknown"

        if is_newborn and is_synd:
            # Use original (not lowercased) to match FEAT_* tags reliably
            orig_caption = train_dataset.custom_instance_prompts[i]
            feats = feat_re.findall(orig_caption)
            if not feats:
                feats = ["FEAT_NONE"]  # just in case

            for f in feats:
                feat_race_counts[f][race] += 1

        sample_groups.append(g)
        group_counts[g] += 1

    print("Group counts:", dict(group_counts))

    def print_feat_table(feat_counts: dict, title: str):
        races = sorted({r for c in feat_counts.values() for r in c.keys()})
        print(f"\n{title}")
        header = "feature".ljust(30) + " " + " ".join(r.rjust(10) for r in races) + "   total"
        print(header)
        print("-" * len(header))

        for feat, ctr in sorted(feat_counts.items(), key=lambda kv: sum(kv[1].values()), reverse=True):
            total = sum(ctr.values())
            row = feat.ljust(30) + " " + " ".join(str(ctr.get(r, 0)).rjust(10) for r in races) + f"   {total}"
            print(row)

    print_feat_table(feat_race_counts, "Feature counts by race (NEWBORN + SYNDROMIC ONLY)")


    group_priority = {
        
        "newborn_syndromic_hispanic": 0.0,
        "newborn_syndromic_asian":    4.0,
        "newborn_syndromic_mena":     4.0,
        "newborn_syndromic_white":    4.0,
        "newborn_syndromic_black":    4.0,

        
        "newborn_healthy_hispanic": 0.0,
        "newborn_healthy_asian":    2.0,
        "newborn_healthy_mena":     2.0,
        "newborn_healthy_white":    2.0,
        "newborn_healthy_black":    2.0,

        
        "older_syndromic_asian":    1.0,
        "older_syndromic_mena":     1.0,
        "older_syndromic_white":    1.0,
        "older_syndromic_black":    1.0,
        "older_syndromic_hispanic": 1.0,

        
        "older_healthy_asian":    0.5,
        "older_healthy_mena":     0.5,
        "older_healthy_white":    0.5,
        "older_healthy_black":    0.5,
        "older_healthy_hispanic": 0.5,

        # Unknowns
        "older_healthy_unknown":   0.0,
        "older_syndromic_unknown": 0.0,
    }


    def print_implied_share(sample_groups, weights, title):
        mass = defaultdict(float)
        for g, w in zip(sample_groups, weights):
            mass[g] += float(w)
        total = sum(mass.values()) or 1.0
        print(f"\n{title}")
        for k in sorted(mass.keys()):
            print(f"{k:30s}  mass={mass[k]:.6f}  share={mass[k]/total:.3%}")

    # ----------------------------
    # Weight computation (NO CAP)
    # ----------------------------
    weights = []
    for g in sample_groups:
        base = group_priority.get(g, 0.0)
        count = group_counts[g]
        w = (base / count) if (base > 0.0 and count > 0) else 0.0
        weights.append(w)

    print_implied_share(sample_groups, weights, "Implied sampling share (NO cap):")

    weights_tensor = torch.as_tensor(weights, dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=len(weights_tensor),
        replacement=True,
    )


    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=sampler,                  # use sampler instead of shuffle
        shuffle=False,                    # MUST be False when using sampler
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=0,
    )


    def compute_time_ids():
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        original_size = (size, size)
        target_size = (size, size)
        crops_coords_top_left = (0, 0)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
        return add_time_ids
        
    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    def compute_text_embeddings(prompts, text_encoders, tokenizers, te_frozen: bool):
        if te_frozen:
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompts)
        else:
            prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompts)

        return (prompt_embeds.to(accelerator.device),
               pooled_prompt_embeds.to(accelerator.device))



    # Number of training steps
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )

    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
        num_cycles=lr_num_cycles,
        power=lr_power,
    )

    # Prepare Everything
    unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler
    )

    text_encoder_one.train()
    text_encoder_two.train()
    print("TE1 mode:", text_encoder_one.training)
    print("TE2 mode:", text_encoder_two.training)

    # Recalculate the training steps
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )

    if overrode_max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch

    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    """
    if accelerator.is_main_process:
            accelerator.init_trackers("dreambooth-lora-sd-xl", config=vars(args))
    """

    # Start Training
    total_batch_size = (
        train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    initial_global_step = 0

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Train Loop
    for epoch in range(first_epoch, num_train_epochs):
        unet.train()

        for step, batch in enumerate(train_dataloader):
            
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=vae.dtype)
                prompts = batch["prompts"]  # list[str], length == bs

                bs = pixel_values.shape[0]

                # SDXL time ids must be [bs, 6] typically
                add_time_ids = compute_time_ids().to(accelerator.device).repeat(bs, 1)

                # Compute embeddings for the whole batch (encode_prompt supports list[str])
                prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                    prompts=prompts,
                    text_encoders=text_encoders,
                    tokenizers=tokenizers,
                    te_frozen=_te_frozen,
                )


                unet_add_text_embeds = pooled_prompt_embeds  # shape [bs, 1280] in SDXL

                # Convert image to latents
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor
                model_input = model_input.to(weight_dtype)

                # Sample noise
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                )
                timesteps = timesteps.long()

                # Add noise
                noisy_model_input = noise_scheduler.add_noise(
                    model_input, noise, timesteps
                )

                unet_added_conditions = {
                    "time_ids": add_time_ids,
                    "text_embeds": unet_add_text_embeds,
                }

                model_pred = unet(
                    noisy_model_input,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=unet_added_conditions,
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )
                    
                loss = F.mse_loss(
                    model_pred.float(), target.float(), reduction="mean"
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    # params_to_clip = (unet_lora_parameters)
                    params_to_clip = unet_lora_parameters + te1_trainable + te2_trainable
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if (not _te_frozen) and (global_step >= FREEZE_STEP):
                    if accelerator.is_main_process:
                        print(f"[Two-phase] Freezing text encoders at step {global_step}")

                    # Freeze TE modules (no more gradients)
                    freeze_module_params_(text_encoder_one)
                    freeze_module_params_(text_encoder_two)

                    # Stop updating TE groups by setting lr=0 (keeps optimizer structure stable)
                    set_param_group_lr_(optimizer, 1, 0.0)
                    set_param_group_lr_(optimizer, 2, 0.0)

                    # Optional: put TE into eval mode (UNet can stay train())
                    text_encoder_one.eval()
                    text_encoder_two.eval()

                    # Sanity prints
                    if accelerator.is_main_process:
                        print("TE1 any trainable?", any_trainable(text_encoder_one))
                        print("TE2 any trainable?", any_trainable(text_encoder_two))
                        print("Optimizer group LRs:", [g["lr"] for g in optimizer.param_groups])

                    _te_frozen = True

                if accelerator.is_main_process and global_step % checkpointing_steps == 0:
                    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)

                    # Save PEFT adapters for inference
                    accelerator.unwrap_model(unet).save_pretrained(os.path.join(save_path, "unet_lora"))
                    accelerator.unwrap_model(text_encoder_one).save_pretrained(os.path.join(save_path, "te1_lora"))
                    accelerator.unwrap_model(text_encoder_two).save_pretrained(os.path.join(save_path, "te2_lora"))

                    logger.info(f"Saved checkpoint + PEFT adapters to {save_path}")
            
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= max_train_steps:
                break

    # Save LoRA Layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet_ = accelerator.unwrap_model(unet)
        te1_  = accelerator.unwrap_model(text_encoder_one)
        te2_  = accelerator.unwrap_model(text_encoder_two)

        # Save PEFT adapters (portable + correct format)
        unet_.save_pretrained(os.path.join(output_dir, "unet_lora"))
        te1_.save_pretrained(os.path.join(output_dir, "te1_lora"))
        te2_.save_pretrained(os.path.join(output_dir, "te2_lora"))

        print("Saved PEFT adapters to:")
        print(" ", os.path.join(output_dir, "unet_lora"))
        print(" ", os.path.join(output_dir, "te1_lora"))
        print(" ", os.path.join(output_dir, "te2_lora"))

        # optional cleanup
        del unet_, te1_, te2_
        del optimizer

    accelerator.end_training()

def parse_args():
    parser = argparse.ArgumentParser(description="Training Script Arguments")

    parser.add_argument("--freeze_step", type=int, default=4000)

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )

    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="The path to the folder containing the training data."
    )

    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt identifying the instance, for this task a prompt like 'a baby with [t]' is suitable, where [t] indicates the disease"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="The path to save the model"
    )

    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training"
    )

    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="The resolution for input images, all the images in the train dataset will be resized to this resolution"
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many times to repeat the training data"
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=64,
        help="The dimension of the LoRA matrices"
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period)"
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="The scheduler type to use. Choose between Choose between ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']"
    )

    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler."
    )
    
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1000,
        help="Total number of training steps to perform"
    )

    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        )
    )

    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass."
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass."
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW"
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        )
    )

    args = parser.parse_args()

    return args



if __name__ == "__main__":
    args = parse_args()

    main(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        instance_data_dir=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        output_dir=args.output_dir,
        size=args.size, repeats=args.repeats,
        rank=args.rank,
        train_batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        max_train_steps=args.max_train_steps,
        checkpointing_steps=args.checkpointing_steps,
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_8bit_adam=args.use_8bit_adam,
        mixed_precision=args.mixed_precision
    )
