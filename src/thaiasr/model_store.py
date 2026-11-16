"""Downloading and verifying the ONNX model files.

The repository keeps the encoder *graph* inside a stored zip (the weights are
already on the Hub as ``encoder*.onnx.data``), so a download is: fetch the zip,
fetch the weights, extract the zip next to the weights, check the sizes.

Nothing here needs PyTorch or NeMo -- just ``huggingface_hub``.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

__all__ = ["REPO_ID", "PRECISIONS", "download_model", "model_ready", "verify_model", "model_size_hint"]

REPO_ID = "iapp/OpenThai2.0-ASR-Streaming-ONNX"

PRECISIONS: dict[str, dict] = {
    "fp32": {
        "zip": "openthai2.0-asr-streaming-fp32.asr.zip",
        "zip_size": 149_365_631,
        "data": "encoder.onnx.data",
        "data_size": 2_454_405_120,
        "members": {
            "encoder.onnx": 41_941_207,
            "decoder_joint.onnx": 107_330_778,
            "tokens.txt": 93_160,
        },
        "total": 2_603_770_751,
    },
    "int8": {
        "zip": "openthai2.0-asr-streaming-int8.asr.zip",
        "zip_size": 109_014_005,
        "data": "encoder.int8.onnx.data",
        "data_size": 883_175_424,
        "members": {
            "encoder.int8.onnx": None,  # size varies; checked as "present"
            "decoder_joint.onnx": 107_330_778,
            "tokens.txt": 93_160,
        },
        "total": 992_189_429,
    },
}


def model_size_hint(precision: str = "fp32") -> str:
    info = PRECISIONS[precision]
    return f"{info['total'] / 1e9:.2f} GB"


def _check_sizes(model_dir: Path, precision: str) -> list[str]:
    """Return a list of problems (empty when everything looks right)."""
    info = PRECISIONS[precision]
    problems: list[str] = []
    for name, expected in info["members"].items():
        path = model_dir / name
        if not path.exists():
            problems.append(f"missing {name}")
        elif expected is not None and path.stat().st_size != expected:
            problems.append(
                f"{name} has {path.stat().st_size} bytes, expected {expected}"
            )
    data = model_dir / info["data"]
    if not data.exists():
        problems.append(f"missing {info['data']}")
    elif data.stat().st_size != info["data_size"]:
        problems.append(
            f"{info['data']} has {data.stat().st_size} bytes, expected {info['data_size']}"
        )
    return problems


def verify_model(model_dir: str | Path, precision: str = "fp32") -> list[str]:
    """Check that the files for ``precision`` are present and the right size."""
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}")
    return _check_sizes(Path(model_dir), precision)


def model_ready(model_dir: str | Path, precision: str = "fp32") -> bool:
    return not verify_model(model_dir, precision)


def download_model(
    model_dir: str | Path,
    *,
    precision: str = "fp32",
    keep_zip: bool = False,
    force: bool = False,
    progress=print,
) -> Path:
    """Fetch, extract and verify the model files. Returns the model directory."""
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}; use fp32 or int8")
    from huggingface_hub import hf_hub_download

    info = PRECISIONS[precision]
    target = Path(model_dir)
    target.mkdir(parents=True, exist_ok=True)

    if not force and not _check_sizes(target, precision):
        progress(f"model already present and verified in {target}")
        return target

    progress(f"downloading {precision} model ({model_size_hint(precision)}) into {target}")
    progress(f"  1/3 {info['zip']} ({info['zip_size'] / 1e6:.0f} MB)")
    zip_path = Path(
        hf_hub_download(REPO_ID, info["zip"], local_dir=str(target), force_download=force)
    )
    progress(f"  2/3 {info['data']} ({info['data_size'] / 1e9:.2f} GB)")
    hf_hub_download(REPO_ID, info["data"], local_dir=str(target), force_download=force)

    progress("  3/3 extracting the graphs")
    with zipfile.ZipFile(zip_path) as zf:
        for member, expected in info["members"].items():
            dest = target / member
            if (
                not force
                and expected is not None
                and dest.exists()
                and dest.stat().st_size == expected
            ):
                continue
            with zf.open(member) as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
    if not keep_zip:
        zip_path.unlink(missing_ok=True)

    problems = _check_sizes(target, precision)
    if problems:
        raise RuntimeError("model files failed verification:\n  " + "\n  ".join(problems))
    progress(f"ready: {target}")
    return target
