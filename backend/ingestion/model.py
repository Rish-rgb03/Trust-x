"""
TRUST-X model ingestion / adapter layer (Person 2).

This module keeps model-loading details separate from model forensics.
The forensic code works with a small adapter interface instead of
depending directly on a particular ML framework.

v0.1 supports:
- metadata extraction for files registered in the TRUST-X DB
- optional PyTorch loading when torch is installed
- a callable predictor adapter for tests/custom runtimes
- simple structural fingerprint extraction
- JSON-safe prediction conversion

For real production models, prefer an explicit adapter around the
framework used by the project (PyTorch/ONNX/etc.) rather than assuming
every model file can be loaded safely.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from ..provenance.hashing import sha256_file


@dataclass
class ModelMetadata:
    path: str
    sha256: str
    format: str
    name: str
    parameter_count: Optional[int] = None
    layer_count: Optional[int] = None
    input_shape: Optional[tuple[int, ...]] = None
    architecture: Optional[str] = None


class ModelAdapter:
    """Small interface used by behavioral.py and triggers.py.

    `predict_fn` receives a PIL image and returns any prediction object.
    This makes the forensic layer easy to test with a fake model and
    keeps it independent of a specific CV library.
    """

    def __init__(
        self,
        predict_fn: Callable[[Any], Any],
        metadata: ModelMetadata,
        raw_model: Any = None,
    ):
        self._predict_fn = predict_fn
        self.metadata = metadata
        self.raw_model = raw_model

    def predict(self, image: Any) -> Any:
        return self._predict_fn(image)


def detect_model_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".pt": "pytorch",
        ".pth": "pytorch",
        ".jit": "torchscript",
        ".onnx": "onnx",
        ".safetensors": "safetensors",
    }.get(suffix, suffix.lstrip(".") or "unknown")


def _count_parameters(model: Any) -> Optional[int]:
    try:
        return sum(int(p.numel()) for p in model.parameters())
    except Exception:
        return None


def _count_layers(model: Any) -> Optional[int]:
    try:
        # named_modules includes the root module, so exclude it.
        return sum(1 for _name, _module in model.named_modules()) - 1
    except Exception:
        return None


def _architecture_name(model: Any) -> Optional[str]:
    try:
        return model.__class__.__name__
    except Exception:
        return None


def extract_structural_metadata(
    model: Any,
    *,
    path: str | Path,
    input_shape: Optional[tuple[int, ...]] = None,
) -> ModelMetadata:
    """Build a structural fingerprint for an already-loaded model."""
    path = str(path)
    return ModelMetadata(
        path=path,
        sha256=sha256_file(path),
        format=detect_model_format(path),
        name=Path(path).stem,
        parameter_count=_count_parameters(model),
        layer_count=_count_layers(model),
        input_shape=input_shape,
        architecture=_architecture_name(model),
    )


def load_pytorch_adapter(
    path: str | Path,
    *,
    device: str = "cpu",
    input_shape: Optional[tuple[int, int, int]] = (3, 224, 224),
) -> ModelAdapter:
    """Load a TorchScript or serialized PyTorch model.

    This is intentionally opt-in: PyTorch is not pinned in requirements.txt
    because it is a large, platform-specific dependency. Install the
    appropriate PyTorch build on machines that need .pt/.pth support.

    The loader accepts:
      1. TorchScript files via torch.jit.load
      2. Serialized nn.Module objects via torch.load

    State-dict-only files are rejected because architecture information is
    unavailable without the model definition.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install the appropriate PyTorch build "
            "to load .pt/.pth models."
        ) from exc

    path = str(path)
    suffix = Path(path).suffix.lower()

    if suffix == ".jit":
        model = torch.jit.load(path, map_location=device)
    else:
        # weights_only=False is required for serialized nn.Module objects.
        # Only load model files you trust; PyTorch deserialization can execute
        # arbitrary code embedded in a pickle.
        model = torch.load(path, map_location=device, weights_only=False)

    if not callable(model):
        raise ValueError(
            "The loaded PyTorch artifact is not callable. A state_dict alone "
            "is not enough for model-agnostic inference."
        )

    model.eval()
    metadata = extract_structural_metadata(
        model, path=path, input_shape=input_shape
    )

    def predict(image):
        # Import PIL lazily so this module remains easy to import.
        import numpy as np

        if hasattr(image, "convert"):
            rgb = image.convert("RGB")
        else:
            rgb = image

        arr = np.asarray(rgb).astype("float32") / 255.0
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("Expected an RGB image with shape HxWx3.")

        # Resize to the declared model input size.
        from PIL import Image

        _, height, width = input_shape
        resized = Image.fromarray((arr * 255).astype("uint8")).resize(
            (width, height)
        )
        arr = np.asarray(resized).astype("float32") / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor.to(device))
        return to_jsonable(output)

    return ModelAdapter(predict, metadata, raw_model=model)


def make_callable_adapter(
    predict_fn: Callable[[Any], Any],
    *,
    name: str = "callable_model",
    input_shape: Optional[tuple[int, ...]] = None,
    architecture: str = "callable",
    parameter_count: Optional[int] = None,
    layer_count: Optional[int] = None,
) -> ModelAdapter:
    """Create an adapter around an existing predictor.

    Useful for:
    - unit tests
    - an existing inference server
    - a custom YOLO/ONNX wrapper
    - a black-box model API
    """
    metadata = ModelMetadata(
        path="",
        sha256="",
        format="black_box",
        name=name,
        parameter_count=parameter_count,
        layer_count=layer_count,
        input_shape=input_shape,
        architecture=architecture,
    )
    return ModelAdapter(predict_fn, metadata)


def to_jsonable(value: Any) -> Any:
    """Convert common ML outputs into JSON-safe Python values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]

    # numpy arrays/scalars
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass

    # torch tensors
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except ImportError:
        pass

    # Dataclasses / common result objects
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass

    if hasattr(value, "model_dump"):
        try:
            return to_jsonable(value.model_dump())
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            return {
                str(k): to_jsonable(v)
                for k, v in vars(value).items()
                if not k.startswith("_")
            }
        except Exception:
            pass

    # Last resort: preserve a useful textual representation rather than
    # crashing the forensic run.
    return str(value)


def structural_difference(
    reference: ModelMetadata,
    candidate: ModelMetadata,
) -> dict[str, Any]:
    """Compare two structural fingerprints and return only observed differences."""
    differences: dict[str, Any] = {}

    fields = (
        "format",
        "architecture",
        "parameter_count",
        "layer_count",
        "input_shape",
    )
    for field in fields:
        a = getattr(reference, field)
        b = getattr(candidate, field)
        if a != b and (a is not None or b is not None):
            differences[field] = {"reference": a, "candidate": b}

    return differences
