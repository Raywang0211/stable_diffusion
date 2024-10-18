import torch
from diffusers import StableDiffusion3Pipeline
from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

model_name = "C:\\Users\\ray.wang\\Desktop\\project\\stable-diffusion-3-medium-diffusers"
# 指定 Hugging Face 上的模型名稱
# model_id = "stabilityai/stable-diffusion-3-medium-diffusers"

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

# # 從 Hugging Face 下載並加載模型
pipe = StableDiffusion3Pipeline.from_pretrained(model_name, torch_dtype=torch.float16, quantization_config=quantization_config)
pipe.enable_model_cpu_offload()

image = pipe(
    prompt="a girl with short hair and a red dress. Big eyes and a small mouth. She is smiling.",
    negative_prompt="",
    num_inference_steps=28,
    height=1024,
    width=1024,
    guidance_scale=7.0,
).images[0]

image.save("sd3_hello_world.png")
# pipe = pipe.to("cuda")  # 將模型加載到 GPU


# tokenizer = AutoModelForCausalLM.from_pretrained(model_name)