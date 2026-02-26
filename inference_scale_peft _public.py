import torch
import random
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from peft import PeftModel
from torchvision.utils import save_image
import torchvision.transforms as T
from pathlib import Path
import re

pipe = StableDiffusionXLPipeline.from_pretrained(
    "path to pretrained_model",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

model_path = Path(
    "path to trained LoRA checkpoint"
)

pipe.unet = PeftModel.from_pretrained(pipe.unet, model_path / "unet_lora").to("cuda")
pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, model_path / "te1_lora").to("cuda")
pipe.text_encoder_2 = PeftModel.from_pretrained(pipe.text_encoder_2, model_path / "te2_lora").to("cuda")

alpha = 1.0
pipe.unet.set_adapter("default")
pipe.text_encoder.set_adapter("default")
pipe.text_encoder_2.set_adapter("default")
pipe.set_adapters(["default"], adapter_weights=[alpha])

out_root = model_path / f"path to output folder"
out_root.mkdir(parents=True, exist_ok=True)

RACES = [
    ("white", "White"),
    ("black", "Black or African American"),
    ("asian", "Asian"),
    ("mena", "Middle Eastern or North African"),
]

RACE_PLAN = {
    "white": {"start_i": 0, "n_images": 200},  
    "black": {"start_i": 0, "n_images": 200},  
    "asian": {"start_i": 0, "n_images": 200},    
    "mena":  {"start_i": 0, "n_images": 200},   
}


FEAT_TEMPLATE = (
    "smartphone photo of a {race} newborn with a genetic syndrome, {feat}, "
    "close-up face, neutral background, not retouched, full color, natural skin tones"
)

neg = (
    "pink skin, magenta cast, red tint, oversaturated, unnatural skin tone, "
    "beauty retouch, airbrushed, plastic skin, "
    "black and white, monochrome, grayscale, desaturated, low saturation, colorless"
)

FEATURES = [
    "upslanting eyes",
    "long philtrum",
    "triangular-shaped mouth",
    "full cheeks",
    "skin folds from the upper eyelid covering the inner corner of the eye",
    "upturned nose tip",
    "an upward fold from the lower eyelid toward the inner corner of the eye",
    "soft face with relaxed muscle tone"
]

def slugify(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")[:80]

WIDTH = HEIGHT = 512
STEPS = 30
GUIDANCE = 5.0
trans = T.ToTensor()

RACE_OFFSETS = {k: ridx * 1_000_000_000 for ridx, (k, _) in enumerate(RACES)}
FEATURE_OFFSETS = {feat: fidx * 1_000_000 for fidx, feat in enumerate(FEATURES)}

pipe.set_progress_bar_config(disable=False)


with torch.inference_mode():
    for race_key, race_str in RACES:
        plan = RACE_PLAN.get(race_key, {"start_i": 0, "n_images": 200})
        start_i = plan["start_i"]
        end_i = start_i + plan["n_images"]

        for feat in FEATURES:
            feat_slug = slugify(feat)
            out_feat = out_root / "feat" / feat_slug / race_key
            out_feat.mkdir(parents=True, exist_ok=True)

            for i in range(start_i, end_i):
                seed = RACE_OFFSETS[race_key] + FEATURE_OFFSETS[feat] + i
                out_path = out_feat / f"syn_{race_key}_{feat_slug}_seed{seed}.png"
                if out_path.exists():
                    continue

                feat_prompt = FEAT_TEMPLATE.format(race=race_str, feat=feat)
                gen = torch.Generator(device="cuda").manual_seed(seed)

                img = pipe(
                    prompt=feat_prompt,
                    negative_prompt=neg,
                    generator=gen,
                    width=WIDTH,
                    height=HEIGHT,
                    num_inference_steps=STEPS,
                    guidance_scale=GUIDANCE,
                ).images[0]

                save_image(trans(img), out_path)

        print(f"Done: {race_key}")


print(f"All images saved under: {out_root}")

