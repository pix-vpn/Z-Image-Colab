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
  mode_tag = {"text2img": "t2i", "img2img": "i2i", "inpaint": "inp", "tile_enhance": "tile"}.get(mode, "t2i")
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

def split_into_tiles(image, tile_size, overlap):
    """Split a PIL image into overlapping square tiles.

    Returns a list of (x, y, tile_pil) tuples. Stride = tile_size - overlap.
    The last row/column is shifted so the tiles always cover the full canvas.
    """
    W, H = image.size
    stride = max(1, tile_size - overlap)
    tiles = []
    ys = list(range(0, max(1, H - overlap), stride))
    xs = list(range(0, max(1, W - overlap), stride))
    for y in ys:
        for x in xs:
            x_end = min(x + tile_size, W)
            y_end = min(y + tile_size, H)
            x_start = max(0, x_end - tile_size)
            y_start = max(0, y_end - tile_size)
            tile = image.crop((x_start, y_start, x_end, y_end))
            tiles.append((x_start, y_start, tile))
    return tiles

def make_linear_weight_map(width, height, overlap):
    """Build a (height, width) float32 weight map with linear feathering on the
    `overlap` pixels of each edge. Edges that fall on the outer boundary of
    the overall image (no neighbouring tile) keep weight=1 across the whole
    side. `overlap<=0` returns an all-ones map.
    """
    if overlap <= 0:
        return np.ones((height, width), dtype=np.float32)
    w = np.ones((height, width), dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, overlap + 2)[1:-1]  # exclude 0 and 1 endpoints
    # left edge
    w[:, :overlap] *= ramp[None, :]
    # right edge
    w[:, -overlap:] *= ramp[None, ::-1]
    # top edge
    w[:overlap, :] *= ramp[:, None]
    # bottom edge
    w[-overlap:, :] *= ramp[::-1, None]
    return w

def merge_tiles(tiles, original_size, tile_size, overlap):
    """Re-assemble a list of (x, y, tile_pil) into a single image of
    `original_size` using the linear weight map for blending overlapping regions.
    """
    W, H = original_size
    accum = np.zeros((H, W, 3), dtype=np.float32)
    weight = np.zeros((H, W, 1), dtype=np.float32)
    for x, y, tile in tiles:
        tw, th = tile.size
        tile_arr = np.array(tile.convert("RGB")).astype(np.float32) / 255.0
        wm = make_linear_weight_map(tw, th, overlap).reshape(th, tw, 1)
        accum[y:y + th, x:x + tw] += tile_arr * wm
        weight[y:y + th, x:x + tw] += wm
    merged = accum / np.maximum(weight, 1e-6)
    return Image.fromarray((merged * 255).clip(0, 255).astype(np.uint8))

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

    # tile_enhance 分支: 提前 return, 不走主 KSampler/VAEDecode 链路
    if mode == "tile_enhance":
        if input_image is None:
            raise ValueError("4K 分块重绘模式需要先上传 4K 图片")
        img = input_image.convert("RGB")
        W, H = img.size
        tile_size = int(values.get("tile_size", 768))
        overlap = int(values.get("tile_overlap", 64))
        tiles = split_into_tiles(img, tile_size, overlap)
        rendered_tiles = []
        for idx, (tx, ty, tile) in enumerate(tiles):
            img_tensor = pil_to_comfy_image(tile)
            latent = VAEEncode.encode(vae, img_tensor)[0]
            samples = KSampler.sample(
                unet, seed + idx, steps, cfg, sampler_name, scheduler,
                positive, negative, latent, denoise=denoise,
            )[0]
            decoded = VAEDecode.decode(vae, samples)[0].detach()
            rendered = Image.fromarray(np.array(decoded * 255, dtype=np.uint8)[0])
            rendered_tiles.append((tx, ty, rendered))
            if on_progress is not None:
                on_progress(idx + 1, len(tiles), (tx, ty, rendered))
        merged = merge_tiles(rendered_tiles, (W, H), tile_size, overlap)
        save_path = get_save_path(positive_prompt, mode=mode)
        merged.save(save_path, quality=95)
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
    decoded = VAEDecode.decode(vae, samples)[0].detach()
    save_path=get_save_path(positive_prompt, mode=mode)
    Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0]).save(save_path)
    drive_path="/content/gdrive/MyDrive/z_image_turbo"
    if os.path.exists(drive_path):
        shutil.copy(save_path,drive_path)
    return save_path,seed

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
    tile_size,
    tile_overlap,
    batch_size=1,
    sampler_name="euler",
    scheduler="simple"
):
    image, mask = extract_image_and_mask(input_image_editor)

    if mode in ("img2img", "inpaint", "tile_enhance") and image is not None:
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
            "tile_size": int(tile_size),
            "tile_overlap": int(tile_overlap),
        }
    }

    if mode == "tile_enhance":
        progress = gr.Progress()
        def on_progress(idx, total, tile_info):
            progress(idx / total, desc=f"分块 {idx}/{total}")
        image_path, used_seed = generate(input_data, on_progress=on_progress)
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
              ("4K 分块重绘 (tile_enhance)", "tile_enhance"),
          ],
          value="text2img",
          label="生成模式",
      )

      positive = gr.Textbox(DEFAULT_POSITIVE, label="正向提示词 Positive Prompt", lines=5)

      input_image_editor = gr.ImageEditor(
          type="pil",
          label="输入图片 (图生图直接上传;局部重绘上传后在白色区域涂抹蒙版;4K 分块重绘上传大图)",
          height=400,
          visible=False,
          brush=gr.Brush(colors=["#ffffff"], default_size=50, color_mode="fixed"),
      )

      with gr.Row(visible=False) as tile_settings:
        tile_size = gr.Dropdown([512, 768, 1024], value=768, label="分块尺寸 Tile Size (像素)")
        tile_overlap = gr.Slider(0, 128, value=64, step=8, label="重叠像素 Overlap (推荐 64)")

      with gr.Row():
        aspect = gr.Dropdown(ASPECTS, value="1024x1024 (1:1)", label="宽高比 (图生图/重绘/4K 分块模式下将使用输入图片尺寸)")
        seed = gr.Number(value=0, label="种子 (0 = random)", precision=0)
        steps = gr.Slider(4, 25, value=9, step=1, label="步数 Steps")
      with gr.Row():
        run = gr.Button('🚀 Generate', variant='primary')
      with gr.Accordion('Image Settings', open=False):
        with gr.Row():
          cfg = gr.Slider(0.5, 4.0, value=1.0, step=0.1, label="CFG")
          denoise = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="重绘幅度 Denoise (文生图=1.0;图生图/重绘建议 0.5-0.7;4K 分块建议 0.2-0.4)")
        with gr.Row():
          negative = gr.Textbox(DEFAULT_NEGATIVE, label="反向提示词 Negative Prompt", lines=3)
    with gr.Column():
        download_image=gr.File(label="下载图片 Download Image")
        output_img = gr.Image(label="生成结果 Generated Image", height=480)
        used_seed = gr.Textbox(label="使用的种子 Seed Used", interactive=False,show_copy_button=True)

    def on_mode_change(selected_mode):
        is_img_based = selected_mode in ("img2img", "inpaint")
        is_tile = selected_mode == "tile_enhance"
        show_editor = is_img_based or is_tile
        if is_tile:
            new_denoise = 0.3
        elif is_img_based:
            new_denoise = 0.6
        else:
            new_denoise = 1.0
        return (
            gr.update(visible=show_editor),
            gr.update(visible=is_tile),
            gr.update(value=new_denoise),
        )

    mode.change(
        fn=on_mode_change,
        inputs=[mode],
        outputs=[input_image_editor, tile_settings, denoise],
    )

    run.click(
        fn=generate_ui,
        inputs=[mode, positive, negative, aspect, input_image_editor, seed, steps, cfg, denoise, tile_size, tile_overlap,],
        outputs=[download_image,output_img, used_seed]
    )

demo.launch(share=True, debug=True)
