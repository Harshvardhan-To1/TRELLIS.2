import os
import time
from contextlib import contextmanager
from typing import Literal, Optional
from uuid import uuid4

import fal
from fal.container import ContainerImage
from fal.toolkit import FAL_PERSISTENT_DIR, File
from pydantic import BaseModel, Field


docker_string = r"""
FROM falai/base:3.11-12.4.0

USER root

ENV PYTHONPATH="/app"
ENV OPENCV_IO_ENABLE_OPENEXR="1"
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ENV SPARSE_CONV_BACKEND="flex_gemm"
ENV ATTN_BACKEND="flash_attn"
ENV CUDA_HOME="/usr/local/cuda"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Core deps (match repo guidance: torch 2.6 / cu124)
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0

# Python deps used by TRELLIS.2 + postprocessing
RUN pip install --no-cache-dir \
    "numpy<2" \
    pillow \
    opencv-python-headless \
    imageio imageio-ffmpeg \
    tqdm easydict \
    transformers accelerate huggingface_hub safetensors \
    rembg onnxruntime \
    requests \
    ninja

# CUDA extensions required by TRELLIS.2 / O-Voxel pipeline
RUN pip install --no-cache-dir flash-attn==2.7.3
RUN pip install --no-cache-dir --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0"
RUN pip install --no-cache-dir --no-build-isolation "git+https://github.com/JeffreyXiang/FlexGEMM.git"
RUN pip install --no-cache-dir --no-build-isolation "git+https://github.com/JeffreyXiang/CuMesh.git"

# Install bundled O-Voxel package (used for GLB export)
RUN pip install --no-cache-dir --no-build-isolation /app/o-voxel

# Optional utilities used by the project
RUN pip install --no-cache-dir \
    "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

"""


@contextmanager
def timed():
    start = time.time()
    try:
        yield lambda: time.time() - start
    finally:
        pass


def read_image_from_url(url: str):
    import io
    import requests
    from PIL import Image

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))


def resize_by_preserving_aspect_ratio(image, max_height: int, max_width: int):
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    w, h = image.size
    if w <= 0 or h <= 0:
        return image
    scale = min(max_width / w, max_height / h, 1.0)
    if scale >= 1.0:
        return image
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


class InputModel(BaseModel):
    image_url: str = Field(
        description="URL of the input image to convert to 3D",
        examples=["https://storage.googleapis.com/falserverless/model_tests/video_models/front.png"],
    )

    seed: Optional[int] = Field(description="Random seed for reproducibility", default=None)

    resolution: Literal[512, 1024, 1536] = Field(
        description="Output resolution; higher is slower but more detailed",
        default=1024,
    )

    # Stage 1: sparse structure
    ss_guidance_strength: float = Field(default=7.5, ge=0.0, le=10.0)
    ss_guidance_rescale: float = Field(default=0.7, ge=0.0, le=1.0)
    ss_sampling_steps: int = Field(default=12, ge=1, le=50)
    ss_rescale_t: float = Field(default=5.0, ge=1.0, le=6.0)

    # Stage 2: shape generation
    shape_slat_guidance_strength: float = Field(default=7.5, ge=0.0, le=10.0)
    shape_slat_guidance_rescale: float = Field(default=0.5, ge=0.0, le=1.0)
    shape_slat_sampling_steps: int = Field(default=12, ge=1, le=50)
    shape_slat_rescale_t: float = Field(default=3.0, ge=1.0, le=6.0)

    # Stage 3: texture/material generation
    tex_slat_guidance_strength: float = Field(default=1.0, ge=0.0, le=10.0)
    tex_slat_guidance_rescale: float = Field(default=0.0, ge=0.0, le=1.0)
    tex_slat_sampling_steps: int = Field(default=12, ge=1, le=50)
    tex_slat_rescale_t: float = Field(default=3.0, ge=1.0, le=6.0)

    # Export params
    decimation_target: int = Field(
        description="Target vertex count for mesh simplification during export",
        default=500_000,
        ge=100_000,
        le=2_000_000,
    )
    texture_size: Literal[1024, 2048, 4096] = Field(
        description="Texture resolution",
        default=2048,
    )
    remesh: bool = Field(description="Run remeshing (slower; often improves topology)", default=True)
    remesh_band: float = Field(default=1.0, ge=0.0, le=4.0)
    remesh_project: float = Field(default=0.0, ge=0.0, le=1.0)


class MultiImageInputModel(InputModel):
    image_urls: list[str] = Field(
        description="Multiple views of the same object. Conditioning is averaged across views.",
        examples=[
            [
                "https://storage.googleapis.com/falserverless/model_tests/video_models/front.png",
                "https://storage.googleapis.com/falserverless/model_tests/video_models/back.png",
                "https://storage.googleapis.com/falserverless/model_tests/video_models/left.png",
            ]
        ],
    )


class ObjectOutput(BaseModel):
    model_glb: File = Field(description="Generated 3D GLB file")
    timings: dict[str, float] = Field(description="Processing timings (seconds)")


def _pipeline_type_from_resolution(resolution: int) -> str:
    if resolution == 512:
        return "512"
    if resolution == 1024:
        return "1024_cascade"
    if resolution == 1536:
        return "1536_cascade"
    raise ValueError(f"Unsupported resolution: {resolution}")


def _reduce_cond(cond: dict) -> dict:
    """
    Reduce multi-image conditioning to a single conditioning by averaging across batch.
    """
    out = dict(cond)
    if "cond" in out and getattr(out["cond"], "ndim", 0) >= 1:
        out["cond"] = out["cond"].mean(dim=0, keepdim=True)
    if "neg_cond" in out and getattr(out["neg_cond"], "ndim", 0) >= 1:
        out["neg_cond"] = out["neg_cond"].mean(dim=0, keepdim=True)
    return out


class Trellis2Fal(
    fal.App,
    keep_alive=300,
    max_concurrency=6,
    min_concurrency=1,
    kind="container",
    image=ContainerImage.from_dockerfile_str(docker_string),
    name="trellis2-image-to-3d",
):  # type: ignore
    machine_type = "GPU-H100"

    def setup(self):
        # Persist caches across cold starts
        os.environ["HF_HOME"] = str(FAL_PERSISTENT_DIR / "hf")
        os.environ["TORCH_HOME"] = str(FAL_PERSISTENT_DIR / "torch")
        os.environ["XDG_CACHE_HOME"] = str(FAL_PERSISTENT_DIR / ".cache")

        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        self.pipeline.cuda()

    def _export_glb(self, mesh, *, decimation_target: int, texture_size: int, remesh: bool, remesh_band: float, remesh_project: float):
        import o_voxel

        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=remesh,
            remesh_band=remesh_band,
            remesh_project=remesh_project,
            use_tqdm=False,
            verbose=False,
        )
        out_path = f"/tmp/trellis2_{uuid4().hex}.glb"
        glb.export(out_path, extension_webp=True)
        return out_path

    def _run_single(self, image, input: InputModel):
        import numpy as np
        import torch

        timings: dict[str, float] = {}

        with timed() as took:
            image = resize_by_preserving_aspect_ratio(image, max_height=1024, max_width=1024)
        timings["prepare"] = took()

        seed = int(input.seed) if input.seed is not None else int(np.random.randint(0, np.iinfo(np.int32).max))
        pipeline_type = _pipeline_type_from_resolution(input.resolution)

        ss_params = {
            "steps": input.ss_sampling_steps,
            "guidance_strength": input.ss_guidance_strength,
            "guidance_rescale": input.ss_guidance_rescale,
            "rescale_t": input.ss_rescale_t,
        }
        shape_slat_params = {
            "steps": input.shape_slat_sampling_steps,
            "guidance_strength": input.shape_slat_guidance_strength,
            "guidance_rescale": input.shape_slat_guidance_rescale,
            "rescale_t": input.shape_slat_rescale_t,
        }
        tex_slat_params = {
            "steps": input.tex_slat_sampling_steps,
            "guidance_strength": input.tex_slat_guidance_strength,
            "guidance_rescale": input.tex_slat_guidance_rescale,
            "rescale_t": input.tex_slat_rescale_t,
        }

        with timed() as took:
            mesh = self.pipeline.run(
                image,
                seed=seed,
                preprocess_image=True,
                sparse_structure_sampler_params=ss_params,
                shape_slat_sampler_params=shape_slat_params,
                tex_slat_sampler_params=tex_slat_params,
                pipeline_type=pipeline_type,
            )[0]
            # nvdiffrast has internal face limits; keep this as a safety cap.
            mesh.simplify(16_777_216)
            torch.cuda.empty_cache()
        timings["generation"] = took()

        with timed() as took:
            out_path = self._export_glb(
                mesh,
                decimation_target=input.decimation_target,
                texture_size=int(input.texture_size),
                remesh=input.remesh,
                remesh_band=input.remesh_band,
                remesh_project=input.remesh_project,
            )
        timings["export"] = took()

        return out_path, timings

    def _run_multi(self, images, input: MultiImageInputModel):
        import numpy as np
        import torch

        timings: dict[str, float] = {}

        with timed() as took:
            images = [resize_by_preserving_aspect_ratio(im, max_height=1024, max_width=1024) for im in images]
        timings["prepare"] = took()

        seed = int(input.seed) if input.seed is not None else int(np.random.randint(0, np.iinfo(np.int32).max))
        pipeline_type = _pipeline_type_from_resolution(input.resolution)

        ss_params = {
            "steps": input.ss_sampling_steps,
            "guidance_strength": input.ss_guidance_strength,
            "guidance_rescale": input.ss_guidance_rescale,
            "rescale_t": input.ss_rescale_t,
        }
        shape_slat_params = {
            "steps": input.shape_slat_sampling_steps,
            "guidance_strength": input.shape_slat_guidance_strength,
            "guidance_rescale": input.shape_slat_guidance_rescale,
            "rescale_t": input.shape_slat_rescale_t,
        }
        tex_slat_params = {
            "steps": input.tex_slat_sampling_steps,
            "guidance_strength": input.tex_slat_guidance_strength,
            "guidance_rescale": input.tex_slat_guidance_rescale,
            "rescale_t": input.tex_slat_rescale_t,
        }

        with timed() as took:
            # Preprocess each view (crop & BG remove if needed).
            proc_images = [self.pipeline.preprocess_image(im) for im in images]
            torch.manual_seed(seed)

            cond_512 = _reduce_cond(self.pipeline.get_cond(proc_images, 512))
            cond_1024 = _reduce_cond(self.pipeline.get_cond(proc_images, 1024)) if pipeline_type != "512" else None

            ss_res = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32}[pipeline_type]
            coords = self.pipeline.sample_sparse_structure(cond_512, ss_res, 1, ss_params)

            if pipeline_type == "512":
                shape_slat = self.pipeline.sample_shape_slat(
                    cond_512, self.pipeline.models["shape_slat_flow_model_512"], coords, shape_slat_params
                )
                tex_slat = self.pipeline.sample_tex_slat(
                    cond_512, self.pipeline.models["tex_slat_flow_model_512"], shape_slat, tex_slat_params
                )
                res = 512
            elif pipeline_type == "1024_cascade":
                shape_slat, res = self.pipeline.sample_shape_slat_cascade(
                    cond_512,
                    cond_1024,
                    self.pipeline.models["shape_slat_flow_model_512"],
                    self.pipeline.models["shape_slat_flow_model_1024"],
                    512,
                    1024,
                    coords,
                    shape_slat_params,
                    49152,
                )
                tex_slat = self.pipeline.sample_tex_slat(
                    cond_1024, self.pipeline.models["tex_slat_flow_model_1024"], shape_slat, tex_slat_params
                )
            elif pipeline_type == "1536_cascade":
                shape_slat, res = self.pipeline.sample_shape_slat_cascade(
                    cond_512,
                    cond_1024,
                    self.pipeline.models["shape_slat_flow_model_512"],
                    self.pipeline.models["shape_slat_flow_model_1024"],
                    512,
                    1536,
                    coords,
                    shape_slat_params,
                    49152,
                )
                tex_slat = self.pipeline.sample_tex_slat(
                    cond_1024, self.pipeline.models["tex_slat_flow_model_1024"], shape_slat, tex_slat_params
                )
            else:
                raise ValueError(f"Multi-image not implemented for pipeline_type={pipeline_type}")

            torch.cuda.empty_cache()
            mesh = self.pipeline.decode_latent(shape_slat, tex_slat, res)[0]
            mesh.simplify(16_777_216)
            torch.cuda.empty_cache()
        timings["generation"] = took()

        with timed() as took:
            out_path = self._export_glb(
                mesh,
                decimation_target=input.decimation_target,
                texture_size=int(input.texture_size),
                remesh=input.remesh,
                remesh_band=input.remesh_band,
                remesh_project=input.remesh_project,
            )
        timings["export"] = took()

        return out_path, timings

    @fal.endpoint("/")
    def generate(self, input: InputModel) -> ObjectOutput:
        with timed() as took:
            image = read_image_from_url(input.image_url)
        dl_time = took()

        out_path, timings = self._run_single(image, input)
        timings = {"download": dl_time, **timings}

        return ObjectOutput(model_glb=File.from_path(out_path), timings=timings)

    @fal.endpoint("/multi")
    def generate_multi(self, input: MultiImageInputModel) -> ObjectOutput:
        with timed() as took:
            images = [read_image_from_url(u) for u in input.image_urls]
        dl_time = took()

        out_path, timings = self._run_multi(images, input)
        timings = {"download": dl_time, **timings}

        return ObjectOutput(model_glb=File.from_path(out_path), timings=timings)


if __name__ == "__main__":
    app_fn = fal.wrap_app(Trellis2Fal)
    app_fn()

