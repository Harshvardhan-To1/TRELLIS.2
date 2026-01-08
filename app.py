import gradio as gr

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from datetime import datetime
import shutil
from typing import *
import torch
import numpy as np
from PIL import Image
from trellis2.modules.sparse import SparseTensor
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel


MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')


css = """
/* Overwrite Gradio Default Style */
.stepper-wrapper {
    padding: 0;
}

.stepper-container {
    padding: 0;
    align-items: center;
}

.step-button {
    flex-direction: row;
}

.step-connector {
    transform: none;
}

.step-number {
    width: 16px;
    height: 16px;
}

.step-label {
    position: relative;
    bottom: 0;
}

.wrap.center.full {
    inset: 0;
    height: 100%;
}

.wrap.center.full.translucent {
    background: var(--block-background-fill);
}

.meta-text-center {
    display: block !important;
    position: absolute !important;
    top: unset !important;
    bottom: 0 !important;
    right: 0 !important;
    transform: unset !important;
}
"""


preview_disabled_html = """
<div style="height: 722px; display: flex; align-items: center; justify-content: center; opacity: 0.7;">
  <div style="max-width: 520px; text-align: center; line-height: 1.4;">
    <p style="margin: 0; font-size: 14px;">
      Preview rendering has been removed. Click <b>Extract GLB</b> to generate a 3D asset for viewing/downloading.
    </p>
  </div>
</div>
"""


def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    
    
def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir)


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess the input image.

    Args:
        image (Image.Image): The input image.

    Returns:
        Image.Image: The preprocessed image.
    """
    processed_image = pipeline.preprocess_image(image)
    return processed_image


def pack_state(latents: Tuple[SparseTensor, SparseTensor, int]) -> dict:
    shape_slat, tex_slat, res = latents
    return {
        'shape_slat_feats': shape_slat.feats.cpu().numpy(),
        'tex_slat_feats': tex_slat.feats.cpu().numpy(),
        'coords': shape_slat.coords.cpu().numpy(),
        'res': res,
    }
    
    
def unpack_state(state: dict) -> Tuple[SparseTensor, SparseTensor, int]:
    shape_slat = SparseTensor(
        feats=torch.from_numpy(state['shape_slat_feats']).cuda(),
        coords=torch.from_numpy(state['coords']).cuda(),
    )
    tex_slat = shape_slat.replace(torch.from_numpy(state['tex_slat_feats']).cuda())
    return shape_slat, tex_slat, state['res']


def get_seed(randomize_seed: bool, seed: int) -> int:
    """
    Get the random seed.
    """
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


def image_to_3d(
    image: Image.Image,
    seed: int,
    resolution: str,
    ss_guidance_strength: float,
    ss_guidance_rescale: float,
    ss_sampling_steps: int,
    ss_rescale_t: float,
    shape_slat_guidance_strength: float,
    shape_slat_guidance_rescale: float,
    shape_slat_sampling_steps: int,
    shape_slat_rescale_t: float,
    tex_slat_guidance_strength: float,
    tex_slat_guidance_rescale: float,
    tex_slat_sampling_steps: int,
    tex_slat_rescale_t: float,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[dict, str]:
    # --- Sampling ---
    _outputs, latents = pipeline.run(
        image,
        seed=seed,
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "guidance_strength": ss_guidance_strength,
            "guidance_rescale": ss_guidance_rescale,
            "rescale_t": ss_rescale_t,
        },
        shape_slat_sampler_params={
            "steps": shape_slat_sampling_steps,
            "guidance_strength": shape_slat_guidance_strength,
            "guidance_rescale": shape_slat_guidance_rescale,
            "rescale_t": shape_slat_rescale_t,
        },
        tex_slat_sampler_params={
            "steps": tex_slat_sampling_steps,
            "guidance_strength": tex_slat_guidance_strength,
            "guidance_rescale": tex_slat_guidance_rescale,
            "rescale_t": tex_slat_rescale_t,
        },
        pipeline_type={
            "512": "512",
            "1024": "1024_cascade",
            "1536": "1536_cascade",
        }[resolution],
        return_latent=True,
    )
    state = pack_state(latents)
    torch.cuda.empty_cache()

    # Keep the second output slot stable, but do not perform any rendering.
    return state, preview_disabled_html


def extract_glb(
    state: dict,
    decimation_target: int,
    texture_size: int,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[str, str]:
    """
    Extract a GLB file from the 3D model.

    Args:
        state (dict): The state of the generated 3D model.
        decimation_target (int): The target face count for decimation.
        texture_size (int): The texture resolution.

    Returns:
        str: The path to the extracted GLB file.
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shape_slat, tex_slat, res = unpack_state(state)
    mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%dT%H%M%S") + f".{now.microsecond // 1000:03d}"
    os.makedirs(user_dir, exist_ok=True)
    glb_path = os.path.join(user_dir, f'sample_{timestamp}.glb')
    glb.export(glb_path, extension_webp=True)
    torch.cuda.empty_cache()
    return glb_path, glb_path


with gr.Blocks(delete_cache=(600, 600)) as demo:
    gr.Markdown("""
    ## Image to 3D Asset with [TRELLIS.2](https://microsoft.github.io/TRELLIS.2)
    * Upload an image (preferably with an alpha-masked foreground object) and click Generate to create a 3D asset.
    * Click Extract GLB to export and download the generated GLB file if you're satisfied with the result. Otherwise, try another time.
    """)
    
    with gr.Row():
        with gr.Column(scale=1, min_width=360):
            image_prompt = gr.Image(label="Image Prompt", format="png", image_mode="RGBA", type="pil", height=400)
            
            resolution = gr.Radio(["512", "1024", "1536"], label="Resolution", value="1024")
            seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
            randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
            decimation_target = gr.Slider(100000, 1000000, label="Decimation Target", value=500000, step=10000)
            texture_size = gr.Slider(1024, 4096, label="Texture Size", value=2048, step=1024)
            
            generate_btn = gr.Button("Generate")
                
            with gr.Accordion(label="Advanced Settings", open=False):                
                gr.Markdown("Stage 1: Sparse Structure Generation")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                    ss_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.7, step=0.01)
                    ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                    ss_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=5.0, step=0.1)
                gr.Markdown("Stage 2: Shape Generation")
                with gr.Row():
                    shape_slat_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                    shape_slat_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.5, step=0.01)
                    shape_slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                    shape_slat_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=3.0, step=0.1)
                gr.Markdown("Stage 3: Material Generation")
                with gr.Row():
                    tex_slat_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance Strength", value=1.0, step=0.1)
                    tex_slat_guidance_rescale = gr.Slider(0.0, 1.0, label="Guidance Rescale", value=0.0, step=0.01)
                    tex_slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                    tex_slat_rescale_t = gr.Slider(1.0, 6.0, label="Rescale T", value=3.0, step=0.1)                

        with gr.Column(scale=10):
            with gr.Walkthrough(selected=0) as walkthrough:
                with gr.Step("Generate", id=0):
                    preview_output = gr.HTML(preview_disabled_html, label="Status", show_label=True, container=True)
                    extract_btn = gr.Button("Extract GLB")
                with gr.Step("Extract", id=1):
                    glb_output = gr.Model3D(label="Extracted GLB", height=724, show_label=True, display_mode="solid", clear_color=(0.25, 0.25, 0.25, 1.0))
                    download_btn = gr.DownloadButton(label="Download GLB")
                    
        with gr.Column(scale=1, min_width=172):
            examples = gr.Examples(
                examples=[
                    f'assets/example_image/{image}'
                    for image in os.listdir("assets/example_image")
                ],
                inputs=[image_prompt],
                fn=preprocess_image,
                outputs=[image_prompt],
                run_on_click=True,
                examples_per_page=18,
            )
                    
    output_buf = gr.State()
    

    # Handlers
    demo.load(start_session)
    demo.unload(end_session)
    
    image_prompt.upload(
        preprocess_image,
        inputs=[image_prompt],
        outputs=[image_prompt],
    )

    generate_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).then(
        lambda: gr.Walkthrough(selected=0), outputs=walkthrough
    ).then(
        image_to_3d,
        inputs=[
            image_prompt, seed, resolution,
            ss_guidance_strength, ss_guidance_rescale, ss_sampling_steps, ss_rescale_t,
            shape_slat_guidance_strength, shape_slat_guidance_rescale, shape_slat_sampling_steps, shape_slat_rescale_t,
            tex_slat_guidance_strength, tex_slat_guidance_rescale, tex_slat_sampling_steps, tex_slat_rescale_t,
        ],
        outputs=[output_buf, preview_output],
    )
    
    extract_btn.click(
        lambda: gr.Walkthrough(selected=1), outputs=walkthrough
    ).then(
        extract_glb,
        inputs=[output_buf, decimation_target, texture_size],
        outputs=[glb_output, download_btn],
    )
        

# Launch the Gradio app
if __name__ == "__main__":
    os.makedirs(TMP_DIR, exist_ok=True)

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
    pipeline.cuda()
    demo.launch(css=css)
