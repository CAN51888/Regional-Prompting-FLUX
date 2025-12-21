import os
os.environ["PYTORCH_NO_NVML"] = "1"  # 避免 NVML 引发断言



import torch
from pipeline_flux_regional import RegionalFluxPipeline, RegionalFluxAttnProcessor2_0
from pipeline_flux_controlnet_regional import RegionalFluxControlNetPipeline
from diffusers import FluxControlNetModel, FluxMultiControlNetModel
# from mask import masks_dict
# import mask
import gc
import torch


if __name__ == "__main__":
        # 如果 mask.py 里面创建了 base_pipe，清掉它
    # if hasattr(mask, "base_pipe"):
    #     try:
    #         mask.base_pipe.to("cpu")
    #     except Exception:
    #         pass
    #     del mask.base_pipe

    # gc.collect()
    # torch.cuda.empty_cache()

    
    model_path = "black-forest-labs/FLUX.1-dev"
    
    use_lora = True
    use_controlnet = False

    if use_controlnet: # takes up more gpu memory
        # READ https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro for detailed usage tutorial
        controlnet_model_union = 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro'
        controlnet_union = FluxControlNetModel.from_pretrained(controlnet_model_union, torch_dtype=torch.bfloat16)
        controlnet = FluxMultiControlNetModel([controlnet_union])
        pipeline = RegionalFluxControlNetPipeline.from_pretrained(model_path, controlnet=controlnet, torch_dtype=torch.bfloat16).to("cuda")
    else:
        pipeline = RegionalFluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
    
    if use_lora:
        # READ https://huggingface.co/Shakker-Labs/FLUX.1-dev-LoRA-Children-Simple-Sketch for detailed usage tutorial
        # pipeline.load_lora_weights("Shakker-Labs/FLUX.1-dev-LoRA-Children-Simple-Sketch", weight_name="FLUX-dev-lora-children-simple-sketch.safetensors")
        pipeline.load_lora_weights("Keltezaa/emma-watson-face-flux-sdxl", weight_name="Emma_Watson_boy40.safetensors")
    
    attn_procs = {}
    for name in pipeline.transformer.attn_processors.keys():
        if 'transformer_blocks' in name and name.endswith("attn.processor"):
            attn_procs[name] = RegionalFluxAttnProcessor2_0()
        else:
            attn_procs[name] = pipeline.transformer.attn_processors[name]
    pipeline.transformer.set_attn_processor(attn_procs)

    masks_dict = torch.load("lora_region_masks.pt")  # load from mask.py output
    # masks_dict = {k: v.to("cpu") for k, v in mask.items}
    # example input with lora enabled
    image_width = 512
    image_height = 512
    num_samples = 1
    num_inference_steps = 24
    guidance_scale = 3.5
    seed = 124
    # base_prompt = "Sketched style: A cute dinosaur playfully blowing tiny fire puffs over a cartoon city in a cheerful scene."
    # base_prompt = "Sketched style: A cute dinosaur is breathing fire, and a cute little cat is watching him curiously."
    base_prompt="a photo of <lora:emma> and <lora:harry> sitting shoulder to shoulder by the fireplace,face lit by its worm glow, detailed, photorealistic"
    background_prompt = "white background"
    regional_prompt_mask_pairs = {
        "0": {
            # "description": "Sketched style: dinosaur with round eyes and a mischievous smile, puffing small flames.",
            "description":"a photo of <lora:emma> sitting by the fireplace, shoulder to shoulder with <lora:harry>, her face softly lit by the warm glow of the fire, detailed, photorealistic, focus on <lora:emma>'s face and expression",
            "mask":  masks_dict['emma']  #[0, 0, 640, 1280]
        },
        "1": {
            # "description": "Sketched style: city with colorful buildings and tiny flames gently floating above, adding a playful touch.", 
            # "description": "Sketched style: A cute cat sitting on a rooftop, curiously watching the dinosaur.",
            "description":"a photo of <lora:harry> sitting by the fireplace, shoulder to shoulder with <lora:emma>, his face softly lit by the warm glow of the fire, detailed, photorealistic, focus on <lora:harry>'s face and expression",
            "mask":  masks_dict['harry']#[640, 0, 1280, 1280]
        }
    }
    ## lora settings
    if use_lora:
        pipeline.fuse_lora(lora_scale=1.5)
    ## region control settings
    mask_inject_steps = 10
    double_inject_blocks_interval = 1 # 18 for full blocks
    single_inject_blocks_interval = 1 # 39 for full blocks
    base_ratio = 0.1

    # prepare regional prompts and masks
    # ensure image width and height are divisible by the vae scale factor
    image_width = (image_width // pipeline.vae_scale_factor) * pipeline.vae_scale_factor
    image_height = (image_height // pipeline.vae_scale_factor) * pipeline.vae_scale_factor

    regional_prompts = []
    regional_masks = []
    background_mask = torch.zeros((image_height, image_width))

    for region_idx, region in regional_prompt_mask_pairs.items():
        description = region['description']
        mask = region['mask']
        # x1, y1, x2, y2 = mask
        mask = mask.to(dtype=torch.int, device="cpu") 
        print("Original mask shape:", tuple(mask.shape))
        # mask = torch.zeros((image_height, image_width))
        # mask[y1:y2, x1:x2] = 1.0

        # background_mask -= mask

        regional_prompts.append(description)
        regional_masks.append(mask)
            
    # if regional masks don't cover the whole image, append background prompt and mask
    if background_mask.sum() > 0:
        regional_prompts.append(background_prompt)
        regional_masks.append(background_mask)

    # setup regional kwargs that pass to the pipeline
    joint_attention_kwargs = {
        'regional_prompts': regional_prompts,
        'regional_masks': regional_masks,
        'double_inject_blocks_interval': double_inject_blocks_interval,
        'single_inject_blocks_interval': single_inject_blocks_interval,
        'base_ratio': base_ratio,
    }

    images = pipeline(
        prompt=base_prompt,
        num_samples=num_samples,
        width=image_width, height=image_height,
        mask_inject_steps=mask_inject_steps,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cuda").manual_seed(seed),
        joint_attention_kwargs=joint_attention_kwargs,
    ).images

    for idx, image in enumerate(images):
        image.save(f"1output_{idx}.jpg")