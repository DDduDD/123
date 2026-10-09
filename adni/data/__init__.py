"""Data utilities for ADNI multimodal pipeline."""

__all__ = [
    "build_manifest",
    "save_manifest",
    "summarize",
]


def __getattr__(name):
    if name in __all__:
        from . import prepare_manifest

        return getattr(prepare_manifest, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
