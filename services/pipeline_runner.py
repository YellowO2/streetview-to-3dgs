"""GPU-wrapped pipeline runners: SHARP/DA3 3DGS generation, DA3-only point
cloud generation, and the Flux image editor. Also owns the ZeroGPU/@spaces.GPU
decorator setup, since that setup exists purely to wrap these calls.

get_pipeline() is the single Pipeline singleton for this whole app --
app.py's Gradio handlers call through here, rather than each loading a
separate copy of the DA3/SHARP models.
"""
import os
import sys
import types

# Stub out pycolmap BEFORE anything can import depth_anything_3 (which
# panoramic_to_3dgs pulls in). depth_anything_3.api unconditionally does
# `from .utils.export import export`, whose __init__ eagerly imports EVERY
# export format handler including .colmap -- which does `import pycolmap`
# at module level, even though we only ever use export_format="mini_npz"
# and never call export_to_colmap ourselves. pycolmap's native _core
# extension has its own pthread_once-guarded static init that touches the
# CUDA driver directly (cudaGetDevice/cuCtxGetDevice_v2), bypassing
# spaces' PyTorch-only .to()/.cuda() emulation entirely -- this is the
# exact native frame in every segfault we've hit chasing this bug (see
# git history/session notes). Since pycolmap is used ONLY inside
# export_to_colmap's function body (never at that file's module level),
# a bare placeholder module satisfies `import pycolmap` without ever
# loading the real native extension, so its problematic init never runs
# in this process at all.
if "pycolmap" not in sys.modules:
    sys.modules["pycolmap"] = types.ModuleType("pycolmap")

try:
    import spaces

    # spaces is also installed locally via requirements.txt, so gate on SPACE_ID
    # which HF Spaces always sets but local machines don't have.
    ON_SPACES = bool(os.getenv("SPACE_ID"))

    if ON_SPACES:
        GPU = spaces.GPU(duration=108)
        GPU_EDIT = spaces.GPU(duration=72)
    else:
        GPU = lambda fn: fn
        GPU_EDIT = lambda fn: fn
except ImportError:
    GPU = lambda fn: fn  # no-op outside HF Spaces
    GPU_EDIT = lambda fn: fn
    ON_SPACES = False

_pipeline = None
_flux_editor = None
if ON_SPACES:
    from editors.flux_editor import FluxEditor
    _flux_editor = FluxEditor(offload=False)


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from panoramic_to_3dgs import Pipeline
        from config import load_pipeline_config
        _pipeline = Pipeline(load_pipeline_config())
    return _pipeline


@GPU_EDIT
def run_editor_gpu(image_path, prompt, mode, output_path):
    global _flux_editor
    if _flux_editor is None:
        from editors.flux_editor import FluxEditor
        _flux_editor = FluxEditor(offload=True)
    _flux_editor.edit(image_path, prompt, mode=mode, output_path=output_path)
    return output_path


@GPU
def run_pipeline_gpu(target_appearance_path, output_dir, scale_mode, gs_backend, support_paths=None, target_depth_path=None):
    pipeline = get_pipeline()
    os.makedirs(output_dir, exist_ok=True)
    pipeline.config.scale_mode = scale_mode
    pipeline.config.gs_backend = gs_backend
    pipeline.run(
        target_appearance_path=target_appearance_path,
        output_dir=output_dir,
        target_depth_path=target_depth_path,
        support_paths=support_paths,
    )

    ply = os.path.join(output_dir, "final_output.ply")
    return ply if os.path.exists(ply) else None


@GPU
def run_pointcloud_gpu(target_depth_path, output_dir, support_paths=None, step_degrees=20):
    pipeline = get_pipeline()
    os.makedirs(output_dir, exist_ok=True)
    pipeline.run_da3_pointcloud(
        target_depth_path=target_depth_path,
        output_dir=output_dir,
        support_paths=support_paths,
        step_degrees=step_degrees,
    )

    ply = os.path.join(output_dir, "da3_pointcloud.ply")
    return ply if os.path.exists(ply) else None
