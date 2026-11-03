"""Live capture and the shared frame-by-frame score-following runtime.

Load the Hydra stage on demand so importing capture or the shared core does not
also import the session orchestrator.
"""

__all__ = ["streaming"]


def __getattr__(name: str):
    if name == "streaming":
        from .runtime import streaming

        return streaming
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
