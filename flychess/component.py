"""Streamlit custom component wrapper for the interactive chess board."""
from pathlib import Path
from typing import Any, Optional
import streamlit.components.v1 as components

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

_COMPONENT: Optional[Any] = None


def _ensure_registered() -> Any:
    """Ensure the custom component is registered in Streamlit's Runtime component registry."""
    global _COMPONENT
    if _COMPONENT is None:
        _COMPONENT = components.declare_component(
            "interactive_board",
            path=str(FRONTEND_DIR)
        )

    # Streamlit requires the component to be registered in the active Runtime's component_registry
    # so that the web server serves the static assets at /component/<name>/index.html
    try:
        from streamlit.runtime import Runtime
        runtime = Runtime.instance()
        if runtime and hasattr(runtime, "component_registry"):
            reg = runtime.component_registry
            if reg.get_component_path(_COMPONENT.name) is None:
                reg.register_component(_COMPONENT)
    except Exception:
        pass

    return _COMPONENT


def interactive_board(*args: Any, **kwargs: Any) -> Any:
    """Invokes the custom component, guaranteeing runtime registration on each execution."""
    comp = _ensure_registered()
    return comp(*args, **kwargs)
