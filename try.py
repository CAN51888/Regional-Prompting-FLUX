import torch
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
).to("cuda")

img = pipe("a cat sitting on a chair", num_inference_steps=28).images[0]
img.save("test_flux.png")
