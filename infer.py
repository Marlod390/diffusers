from diffusers import StableDiffusion3Pipeline
import torch

model_id = "stabilityai/stable-diffusion-3-medium-diffusers"

pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)
pipe = pipe.to("cuda")

prompt = "A cat sitting besides a rocket on a planet with a lot of cactuses"

generator = torch.Generator("cuda").manual_seed(1234)

out = pipe(
    prompt=prompt,
    num_inference_steps=28,

    generator=generator,
    use_vlm_guidance=True,
    debug_vlm=True,
    return_baseline=True,

    save_intermediates=True,
    save_intermediates_dir="sd3_steps",
    save_intermediates_every=1,
    save_x0_predictions=True
)
out["images"][0].save("sd3_guided.png")
out["images_baseline"][0].save("sd3_baseline.png")
