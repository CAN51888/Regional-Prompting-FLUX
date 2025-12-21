import torch
import os
import numpy as np
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from utils import AttnRecorder
from utils import RecordingDoubleStreamProcessor
from utils import compute_lora_masks_from_attn_single_layer
device = "cuda"

# 1. 第一遍：普通 Flux + hook
base_pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
).to(device)

# 2. 挂上 recorder（只关心 step=5, block=16）
recorder = AttnRecorder(target_step=5, target_block=16, device="cpu")
target_block_idx = 16
block = base_pipe.transformer.transformer_blocks[target_block_idx]
block.attn.processor = RecordingDoubleStreamProcessor(recorder=recorder,
                                                      block_idx=target_block_idx)
base_pipe.attn_recorder = recorder

# 3. 一次正常采样（注意：这里不需要任何 regional_mask）
recorder.reset()
# prompt = "<lora:emma>emma and <lora:harry> harry sitting shoulder to shoulder by the fireplace,face lit by its worm glow, detailed, photorealistic"
prompt = "a photo of <lora:emma> and <lora:harry> sitting shoulder to shoulder by the fireplace,face lit by its worm glow, detailed, photorealistic"
tokenizer_2 = base_pipe.tokenizer_2

text_inputs = tokenizer_2(
    prompt,
    padding="max_length",
    max_length=512,
    truncation=True,
    return_tensors="pt",
)
# 这才是真正的 token ids，整型张量 [1, seq_len]
text_token_ids = text_inputs.input_ids.to(device)  # 或者留在 CPU 也行
print("Prompt token ids:", text_token_ids)

image_rough = base_pipe(
    prompt=prompt,
    
    num_inference_steps=20,
    height=512,
    width=512,
    guidance_scale=1.0,
    generator=torch.Generator(device).manual_seed(42),
).images[0]
image_rough_np = np.array(image_rough)

cross = recorder.cross_attn_list[0]   # 让 AttnRecorder 存 CPU 版最好
selfa = recorder.self_attn_list[0]


# activation_words: 自己定义
activation_words = {
    "emma":  ["emma"],
    "harry": ["harry potter"],
} 
H_img, W_img = image_rough_np.shape[:2]
H_latent = H_img // base_pipe.vae_scale_factor
W_latent = W_img // base_pipe.vae_scale_factor
print("Image size:", (H_img, W_img), "Latent size:", (H_latent, W_latent))
masks_dict_cpu = compute_lora_masks_from_attn_single_layer(
    cross_attn=cross,              # CPU tensor
    self_attn=selfa,               # CPU tensor
    text_ids=text_token_ids,
    tokenizer=tokenizer_2,
    activation_words=activation_words,
    image=image_rough_np,
    H_latent=H_latent,
    W_latent=W_latent,
    device="cpu",                  # 显式指定
)

masks_dict = {k: v.to(device) for k, v in masks_dict_cpu.items()}
print("Computed LoRA masks for regions:", masks_dict)

save_path = "lora_region_masks.pt"
os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

# masks_dict_cpu: Dict[str, Tensor]
torch.save(masks_dict, save_path)
print(f"Saved masks to: {save_path}")