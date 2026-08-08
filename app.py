#@title Utils Code
# %cd /content/ComfyUI

import os, random, time, shutil

import torch
import numpy as np
from PIL import Image
import re, uuid
from nodes import NODE_CLASS_MAPPINGS

UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
VAEEncode = NODE_CLASS_MAPPINGS["VAEEncode"]()
VAEEncodeForInpaint = NODE_CLASS_MAPPINGS["VAEEncodeForInpaint"]()

# PiD 节点 (NVIDIA ComfyUI-PiD, https://github.com/Merserk/ComfyUI-PiD)
# 安装: cd ComfyUI/custom_nodes && git clone https://github.com/Merserk/ComfyUI-PiD.git
#      cd ComfyUI-PiD && pip install -r requirements.txt
try:
    PiDUpscale = NODE_CLASS_MAPPINGS["PiDUpscale"]()
    PiDDecode = NODE_CLASS_MAPPINGS["PiDDecode"]()
    PiDCaptionCreator = NODE_CLASS_MAPPINGS["PiDCaptionCreator"]()
    HAS_PID = True
    print("[app] ComfyUI-PiD 已加载 (PiDUpscale / PiDDecode / PiDCaptionCreator)")
except KeyError:
    PiDUpscale = None
    PiDDecode = None
    PiDCaptionCreator = None
    HAS_PID = False
    print("[app] ⚠️ ComfyUI-PiD 未安装,pid_upscale 模式与 PiD 增强开关将不可用")

with torch.inference_mode():
    unet = UNETLoader.load_unet("z-image-turbo-fp8-e4m3fn.safetensors", "fp8_e4m3fn_fast")[0]
    clip = CLIPLoader.load_clip("qwen_3_4b.safetensors", type="lumina2")[0]
    vae = VAELoader.load_vae("ae.safetensors")[0]

save_dir="./results"
os.makedirs(save_dir, exist_ok=True)
def get_save_path(prompt, mode="text2img"):
  save_dir = "./results"
  safe_prompt = re.sub(r'[^a-zA-Z0-9_-]', '_', prompt)[:25]
  uid = uuid.uuid4().hex[:6]
  mode_tag = {"text2img": "t2i", "img2img": "i2i", "inpaint": "inp", "pid_upscale": "pid"}.get(mode, "t2i")
  filename = f"{mode_tag}_{safe_prompt}_{uid}.png"
  path = os.path.join(save_dir, filename)
  return path

def pil_to_comfy_image(pil_image):
    """Convert a PIL.Image (RGB) to a ComfyUI IMAGE tensor (1, H, W, 3) float32 in [0, 1]."""
    arr = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)

def pil_mask_to_comfy_mask(pil_mask):
    """Convert a PIL.Image grayscale mask (255 = masked) to a ComfyUI MASK tensor (1, H, W) float32 in [0, 1]."""
    arr = np.array(pil_mask.convert("L")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)

def extract_image_and_mask(editor_output):
    """Extract (image, mask) from a gr.ImageEditor output dict.

    Returns:
        (PIL.Image or None, PIL.Image grayscale mask or None)
    """
    if editor_output is None:
        return None, None

    if not isinstance(editor_output, dict):
        if isinstance(editor_output, Image.Image):
            return editor_output, None
        return None, None

    background = editor_output.get("background")
    layers = editor_output.get("layers") or []

    if background is None:
        return None, None

    width, height = background.size
    mask_arr = np.zeros((height, width), dtype=np.uint8)
    has_paint = False
    for layer in layers:
        if layer is None:
            continue
        if layer.size != (width, height):
            layer = layer.resize((width, height), Image.LANCZOS)
        layer_rgba = layer.convert("RGBA")
        alpha = np.array(layer_rgba)[:, :, 3]
        painted = (alpha > 0).astype(np.uint8) * 255
        if painted.any():
            has_paint = True
        mask_arr = np.maximum(mask_arr, painted)

    mask_img = Image.fromarray(mask_arr) if has_paint else None
    return background, mask_img

@torch.inference_mode()
def generate(input, on_progress=None):
    values = input["input"]
    mode = values.get("mode", "text2img")
    positive_prompt = values['positive_prompt']
    negative_prompt = values['negative_prompt']
    seed = values['seed'] # 0
    steps = values['steps'] # 9
    cfg = values['cfg'] # 1.0
    sampler_name = values['sampler_name'] # euler
    scheduler = values['scheduler'] # simple
    denoise = values['denoise'] # 1.0
    width = values['width'] # 1024
    height = values['height'] # 1024
    batch_size = values['batch_size'] # 1.0
    input_image = values.get("input_image")
    mask = values.get("mask")

    if seed == 0:
        random.seed(int(time.time()))
        seed = random.randint(0, 18446744073709551615)

    positive = CLIPTextEncode.encode(clip, positive_prompt)[0]
    negative = CLIPTextEncode.encode(clip, negative_prompt)[0]

    # pid_upscale 分支: 提前 return, 不走主 KSampler/VAEDecode 链路
    # 使用 NVIDIA ComfyUI-PiD 节点: 自动生成 caption + 4 步原生 PiD 采样 + 平铺融合
    if mode == "pid_upscale":
        if not HAS_PID:
            raise ValueError(
                "pid_upscale 模式需要先安装 ComfyUI-PiD 节点:\n"
                "  cd ComfyUI/custom_nodes && git clone https://github.com/Merserk/ComfyUI-PiD.git\n"
                "  cd ComfyUI-PiD && pip install -r requirements.txt\n"
                "然后重启 ComfyUI 运行时"
            )
        if input_image is None:
            raise ValueError("PiD 增强重绘模式需要先上传图片")
        img = input_image.convert("RGB")
        img_tensor = pil_to_comfy_image(img)

        # 1) PiD Caption Creator: 用 Qwen3.5-0.8B 从图片生成 caption
        if on_progress is not None:
            on_progress(0, 2, None)  # 阶段 0/2: 生成 caption
        cap_result = PiDCaptionCreator.create(img_tensor, auto_download=True, preview="")
        if isinstance(cap_result, dict):
            _, caption = cap_result["result"]
        else:
            _, caption = cap_result
        caption = (caption or "").strip() or positive_prompt

        # 2) PiD Upscale: 内置分块 + 4 步原生 PiD + 余弦融合 + 缩放到目标倍数
        if on_progress is not None:
            on_progress(1, 2, None)  # 阶段 1/2: PiD 推理
        upscale_factor = str(values.get("pid_upscale_factor", "4x"))
        strength = float(values.get("pid_strength", 0.4))
        upscaled_tensor = PiDUpscale.upscale(
            img_tensor,
            pid_ckpt_type="2kto4k",   # 1024-class → 输出 4x
            version="v1.5",
            backbone="zimage-turbo",  # 与现有 Z-Image-Turbo 模型一致
            auto_download=True,
            model_precision="bf16",
            upscale_factor=upscale_factor,
            strength=strength,
            caption=caption,
        )[0]

        # 3) 保存 (B, H, W, C) float32 [0,1] -> uint8 PNG
        out_W = upscaled_tensor.shape[2]
        out_H = upscaled_tensor.shape[1]
        arr = (upscaled_tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        save_path = get_save_path(positive_prompt, mode=mode)
        Image.fromarray(arr).save(save_path, quality=95)
        drive_path = "/content/gdrive/MyDrive/z_image_turbo"
        if os.path.exists(drive_path):
            shutil.copy(save_path, drive_path)
        return save_path, seed

    if mode == "text2img":
        latent_image = EmptyLatentImage.generate(width, height, batch_size=batch_size)[0]
    elif mode == "img2img":
        if input_image is None:
            raise ValueError("图生图模式需要先上传输入图片")
        img = input_image.convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)
        img_tensor = pil_to_comfy_image(img)
        latent_image = VAEEncode.encode(vae, img_tensor)[0]
    elif mode == "inpaint":
        if input_image is None:
            raise ValueError("局部重绘模式需要先上传输入图片")
        if mask is None:
            raise ValueError("局部重绘模式需要在图片上涂抹蒙版(白色区域)")
        img = input_image.convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)
            mask = mask.resize((width, height), Image.LANCZOS)
        img_tensor = pil_to_comfy_image(img)
        mask_tensor = pil_mask_to_comfy_mask(mask)
        latent_image = VAEEncodeForInpaint.encode(vae, img_tensor, mask_tensor, grow_mask_by=6)[0]
    else:
        raise ValueError(f"未知的生成模式: {mode}")

    samples = KSampler.sample(unet, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)[0]

    # 内联 PiD 解码 (替代 VAEDecode):4 步原生 PiD,细节更锐
    pid_decode_enabled = bool(values.get("pid_decode_enabled", False))
    pid_decode_sigma = float(values.get("pid_decode_sigma", 0.1))
    if pid_decode_enabled and HAS_PID:
        if on_progress is not None:
            on_progress(0, 3, None)
        decoded_tensor = PiDDecode.decode(samples, caption=positive_prompt, sigma=pid_decode_sigma)[0]
    else:
        decoded_tensor = VAEDecode.decode(vae, samples)[0].detach()

    save_path = get_save_path(positive_prompt, mode=mode)
    arr = (decoded_tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(save_path, quality=95)
    drive_path = "/content/gdrive/MyDrive/z_image_turbo"
    if os.path.exists(drive_path):
        shutil.copy(save_path, drive_path)

    # 内联 PiD 超分: 1K → 4K (可选 2x/4x/6x/8x)
    pid_upscale_inline_enabled = bool(values.get("pid_upscale_inline_enabled", False))
    pid_upscale_inline_factor = str(values.get("pid_upscale_inline_factor", "4x"))
    pid_upscale_inline_strength = float(values.get("pid_upscale_inline_strength", 0.4))
    if pid_upscale_inline_enabled and HAS_PID:
        if on_progress is not None:
            on_progress(1, 3, None)
        img_tensor = decoded_tensor  # (1, H, W, 3) float32
        upscaled_tensor = PiDUpscale.upscale(
            img_tensor,
            pid_ckpt_type="2kto4k",
            version="v1.5",
            backbone="zimage-turbo",
            auto_download=True,
            model_precision="bf16",
            upscale_factor=pid_upscale_inline_factor,
            strength=pid_upscale_inline_strength,
            caption=positive_prompt,
        )[0]
        if on_progress is not None:
            on_progress(2, 3, None)
        upscaled_arr = (upscaled_tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        upscaled_path = save_path[:-4] + "_4k.png"
        Image.fromarray(upscaled_arr).save(upscaled_path, quality=95)
        if os.path.exists(drive_path):
            shutil.copy(upscaled_path, drive_path)
        return upscaled_path, seed

    return save_path, seed

import gradio as gr
def generate_ui(
    mode,
    positive_prompt,
    negative_prompt,
    aspect_ratio,
    input_image_editor,
    seed,
    steps,
    cfg,
    denoise,
    pid_upscale_factor,
    pid_strength,
    pid_decode_enabled,
    pid_decode_sigma,
    pid_upscale_inline_enabled,
    pid_upscale_inline_factor,
    pid_upscale_inline_strength,
    batch_size=1,
    sampler_name="euler",
    scheduler="simple"
):
    image, mask = extract_image_and_mask(input_image_editor)

    if mode in ("img2img", "inpaint", "pid_upscale") and image is not None:
        width, height = image.size
    else:
        width, height = [int(x) for x in aspect_ratio.split("(")[0].strip().split("x")]

    input_data = {
        "input": {
            "mode": mode,
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "width": int(width),
            "height": int(height),
            "batch_size": int(batch_size),
            "seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": float(denoise),
            "input_image": image,
            "mask": mask,
            "pid_upscale_factor": str(pid_upscale_factor),
            "pid_strength": float(pid_strength),
            "pid_decode_enabled": bool(pid_decode_enabled),
            "pid_decode_sigma": float(pid_decode_sigma),
            "pid_upscale_inline_enabled": bool(pid_upscale_inline_enabled),
            "pid_upscale_inline_factor": str(pid_upscale_inline_factor),
            "pid_upscale_inline_strength": float(pid_upscale_inline_strength),
        }
    }

    if mode == "pid_upscale":
        progress = gr.Progress()
        def on_progress(stage, total_stages, _tile):
            label = "生成 caption" if stage == 0 else "PiD 推理中"
            progress(stage / max(total_stages, 1), desc=label)
        image_path, used_seed = generate(input_data, on_progress=on_progress)
    elif mode in ("text2img", "img2img", "inpaint"):
        # 内联 PiD 启用时显示进度
        if pid_decode_enabled or pid_upscale_inline_enabled:
            progress = gr.Progress()
            def on_progress(stage, total_stages, _info):
                labels = {0: "PiD 解码中", 1: "PiD 超分中", 2: "保存 4K"}
                progress(stage / max(total_stages, 1), desc=labels.get(stage, "处理中"))
            image_path, used_seed = generate(input_data, on_progress=on_progress)
        else:
            image_path, used_seed = generate(input_data)
    else:
        image_path, used_seed = generate(input_data)
    return image_path, image_path, used_seed



DEFAULT_POSITIVE = """A beautiful woman with platinum blond hair that is almost white, snowy white skin, red bush, very big plump red lips, high cheek bones and sharp. She has almond shaped red eyes and she's holding a intricate mask. She's wearing white and gold royal gown with a black cloak.  In the veins of her neck its gold."""

DEFAULT_NEGATIVE = """low quality, blurry, unnatural skin tone, bad lighting, pixelated,
noise, oversharpen, soft focus,pixelated"""

ASPECTS = [
    "1024x1024 (1:1)", "1152x896 (9:7)", "896x1152 (7:9)",
    "1152x864 (4:3)", "864x1152 (3:4)", "1248x832 (3:2)",
    "832x1248 (2:3)", "1280x720 (16:9)", "720x1280 (9:16)",
    "1344x576 (21:9)", "576x1344 (9:21)"
]


custom_css = ".gradio-container { font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif; }"

with gr.Blocks(theme=gr.themes.Soft(),css=custom_css) as demo:
  gr.HTML("""
<div style=\"width:100%; display:flex; flex-direction:column; align-items:center; justify-content:center; margin:20px 0;\">
    <h1 style=\"font-size:2.5em; margin-bottom:10px;\">Z-Image-Turbo</h1>
    <a href=\"https://github.com/Tongyi-MAI/Z-Image\" target=\"_blank\">
        <img src=\"https://img.shields.io/badge/GitHub-Z--Image-181717?logo=github&logoColor=white\"
             style=\"height:15px;\">
    </a>
</div>
""")


  with gr.Row():
    with gr.Column():
      mode = gr.Radio(
          choices=[
              ("文生图 (txt2img)", "text2img"),
              ("图生图 (img2img)", "img2img"),
              ("局部重绘 (inpaint)", "inpaint"),
              ("PiD 增强重绘 (pid_upscale)", "pid_upscale"),
          ],
          value="text2img",
          label="生成模式",
      )

      positive = gr.Textbox(DEFAULT_POSITIVE, label="正向提示词 Positive Prompt", lines=5)

      input_image_editor = gr.ImageEditor(
          type="pil",
          label="输入图片 (图生图直接上传;局部重绘上传后在白色区域涂抹蒙版;PiD 增强重绘上传需要增强/超分的图片)",
          height=400,
          visible=False,
          brush=gr.Brush(colors=["#ffffff"], default_size=50, color_mode="fixed"),
      )

      with gr.Row(visible=False) as pid_settings:
        pid_upscale_factor = gr.Dropdown(["2x", "4x", "6x", "8x"], value="4x", label="放大倍数 Upscale Factor")
        pid_strength = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="细节强度 Strength (0=保留原图;1=最大重绘)")

      with gr.Row():
        aspect = gr.Dropdown(ASPECTS, value="1024x1024 (1:1)", label="宽高比 (图生图/重绘/PiD 模式下将使用输入图片尺寸)")
        seed = gr.Number(value=0, label="种子 (0 = random)", precision=0)
        steps = gr.Slider(4, 25, value=9, step=1, label="步数 Steps")
      with gr.Row():
        run = gr.Button('🚀 Generate', variant='primary')
      with gr.Accordion('🚀 PiD 增强 (NVIDIA 4步原生扩散, 需先安装 ComfyUI-PiD)', open=False, visible=True) as pid_inline_settings:
        gr.Markdown("**内联叠加** — 与所选模式 (text2img/img2img/inpaint) 组合, 非互斥\n\n- **PiD 解码**: 替代默认 VAE Decode, 4步原生扩散解码, 细节更锐\n- **PiD 超分**: 在生成完成后追加 4x 超分, 一键出 4K (会额外保存 `<name>_4k.png`)")
        with gr.Row():
          pid_decode_enabled = gr.Checkbox(False, label="用 PiD 解码替代 VAE 解码")
          pid_decode_sigma = gr.Slider(0.0, 1.0, value=0.1, step=0.05, label="PiD 解码重绘量 σ (0=接近VAE;1=最大重绘)")
        with gr.Row():
          pid_upscale_inline_enabled = gr.Checkbox(False, label="生成后追加 PiD 超分 (出 4K)")
          pid_upscale_inline_factor = gr.Dropdown(["2x", "4x", "6x", "8x"], value="4x", label="放大倍数")
          pid_upscale_inline_strength = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="超分重绘强度 Strength")
      with gr.Accordion('Image Settings', open=False):
        with gr.Row():
          cfg = gr.Slider(0.5, 4.0, value=1.0, step=0.1, label="CFG")
          denoise = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="重绘幅度 Denoise (文生图=1.0;图生图/重绘建议 0.5-0.7;PiD 模式此项无效)")
        with gr.Row():
          negative = gr.Textbox(DEFAULT_NEGATIVE, label="反向提示词 Negative Prompt", lines=3)
    with gr.Column():
        download_image=gr.File(label="下载图片 Download Image")
        output_img = gr.Image(label="生成结果 Generated Image", height=480)
        used_seed = gr.Textbox(label="使用的种子 Seed Used", interactive=False,show_copy_button=True)

    def on_mode_change(selected_mode):
        is_img_based = selected_mode in ("img2img", "inpaint")
        is_pid = selected_mode == "pid_upscale"
        show_editor = is_img_based or is_pid
        if is_pid:
            new_denoise = 1.0  # PiD 模式不使用 denoise,但保持 slider 可见
        elif is_img_based:
            new_denoise = 0.6
        else:
            new_denoise = 1.0
        return (
            gr.update(visible=show_editor),
            gr.update(visible=is_pid),
            gr.update(value=new_denoise),
            gr.update(visible=not is_pid),  # 内联 PiD 在 pid_upscale 模式下隐藏 (避免概念冲突)
        )

    mode.change(
        fn=on_mode_change,
        inputs=[mode],
        outputs=[input_image_editor, pid_settings, denoise, pid_inline_settings],
    )

    run.click(
        fn=generate_ui,
        inputs=[
            mode, positive, negative, aspect, input_image_editor,
            seed, steps, cfg, denoise,
            pid_upscale_factor, pid_strength,
            pid_decode_enabled, pid_decode_sigma,
            pid_upscale_inline_enabled, pid_upscale_inline_factor, pid_upscale_inline_strength,
        ],
        outputs=[download_image,output_img, used_seed]
    )

demo.launch(share=True, debug=True)
