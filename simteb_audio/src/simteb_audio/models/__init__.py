from .extra_audio_models import register as _register_extra
from .mfcc_baseline import register as _register_mfcc


def register():
    """Register every model MTEB does not ship. Returns the names added."""
    return _register_extra() + _register_mfcc()


__all__ = ["register"]
