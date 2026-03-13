import torch
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from PIL import Image

try:
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny")
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", controlnet=controlnet)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus-face_sd15.bin")

    device = "cpu"
    pipe.to(device)
    face_crop_pil = Image.new("RGB", (224, 224))

    ip_embeds = pipe.prepare_ip_adapter_image_embeds(
        ip_adapter_image=[face_crop_pil],
        ip_adapter_image_embeds=None,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )
    print("Type of ip_embeds:", type(ip_embeds))
    if isinstance(ip_embeds, tuple):
        print("It is a tuple with length", len(ip_embeds))
        for i, item in enumerate(ip_embeds):
            print(f"  {i}: {type(item)}")
    elif isinstance(ip_embeds, list):
        print("It is a list of length", len(ip_embeds))
        for i, item in enumerate(ip_embeds):
            print(f"  {i}: {type(item)}")
            
    # And test passing it
    print("Trying to generate using returned embeds...")
    mask = Image.new("L", (512, 512), 255)
    img = Image.new("RGB", (512, 512))
    canny = Image.new("RGB", (512, 512))
    
    out = pipe(
        prompt="a dog",
        image=img,
        mask_image=mask,
        control_image=canny,
        ip_adapter_image_embeds=ip_embeds,
        num_inference_steps=2,
    )
    print("Generation Success!")
except Exception as e:
    import traceback
    traceback.print_exc()
