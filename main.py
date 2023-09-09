from datetime import datetime
from pathlib import Path
from typing import Optional
import argparse

from PIL import Image
from compel import Compel, ReturnedEmbeddingsType
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
import torch
import torchvision

from clip_vision import load_clipvision_from_sd
from utils.patch import set_timesteps, patched_call
from utils.utils import load_torch_file

EulerDiscreteScheduler.set_timesteps = set_timesteps
StableDiffusionXLPipeline.__call__ = patched_call

def main(
    prompt: Optional[str] = None,
    image_path: str = "./samples/your_name.jpg",
    clip_model_path: str = "./models/clip_vision_g.safetensors",
    sdxl_model_path: str = "./models/sd_xl_base_1.0_0.9vae.safetensors",
    results_folder: str = "./results/",
    seed: Optional[int] = None,
):
    with torch.no_grad():
        # Load CLIP model
        json_config = "./clip_vision_config_g.json"
        sd = load_torch_file(clip_model_path)
        model = load_clipvision_from_sd(sd, json_config)

        # Encode images
        # We can average embeddings from multiple images
        # But we are using one here from simplicity
        image_paths = [image_path]
        embeddings = []
        for path in image_paths:
            image_t = torchvision.io.read_image(path)
            image_t = image_t.permute(1, 2, 0).unsqueeze(0) / 255
            image_t = image_t.to("cuda")
            emb: torch.Tensor = model.encode_image(image_t).image_embeds
            embeddings.append(emb)
        image_embeddings = torch.mean(torch.concat(embeddings, 0), dim=0, keepdim=True)
        model.model.to("cpu")

        # Load SDXL model
        SDXL_MODEL = StableDiffusionXLPipeline.from_single_file(
            sdxl_model_path,
            torch_dtype=torch.float16,
        )
        compel = Compel(
            tokenizer=[SDXL_MODEL.tokenizer, SDXL_MODEL.tokenizer_2],
            text_encoder=[SDXL_MODEL.text_encoder, SDXL_MODEL.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True]
        )
        SDXL_MODEL.enable_model_cpu_offload()

        # Encode prompt
        if prompt:
            prompt_embeds, _ = compel(prompt)
        else:
            # If no prompt is provided, zero out prompt_embeds
            prompt_embeds = torch.zeros((1, 77, 2048))

        # Generate image
        result: Image.Image = SDXL_MODEL(
            prompt_embeds=prompt_embeds * 0.8,
            pooled_prompt_embeds=image_embeddings * 2,
            guidance_scale=7.0,
            num_inference_steps=20,
            generator=None if seed is None else torch.manual_seed(seed),
            sharpness=5.0,
        ).images[0]
        result.save(Path(results_folder) / f"{Path(image_path).stem}_{prompt}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", type=str, default=None)
    parser.add_argument("-i", "--image", type=str, default="./samples/your_name.jpg")
    parser.add_argument("--clip_model", type=str, default="./models/clip_vision_g.safetensors")
    parser.add_argument("--sdxl_model", type=str, default="./models/sd_xl_base_1.0_0.9vae.safetensors")
    parser.add_argument("--out", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    main(args.prompt, args.image, args.clip_model, args.sdxl_model, args.out, args.seed)
