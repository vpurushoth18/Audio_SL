import os

from .extra_audio_models import register as _register_extra
from .mfcc_baseline import register as _register_mfcc
from .expansion_models import register as _register_expansion


def register():
    """Register every model MTEB does not ship. Returns the names added."""
    names = _register_extra() + _register_mfcc() + _register_expansion()
    # Opt-in only: the default path stays identical to the original run.
    if os.environ.get("SIMTEB_CORRECTED_LOADERS") == "1":
        from .corrected_loaders import register_corrected
        fixed = register_corrected()
        if fixed:
            print(f"[corrected-loaders] validating wrapper applied to: "
                  f"{', '.join(fixed)}", flush=True)
    return names


__all__ = ["register"]
