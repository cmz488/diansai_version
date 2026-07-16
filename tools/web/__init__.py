"""OpenCV Web Debugger — engineered debug & display tool."""


def __getattr__(name):
    if name == "DebugServer":
        from tools.web.server import DebugServer as _S
        return _S
    if name == "CameraManager":
        from tools.web.camera import CameraManager as _C
        return _C
    if name == "ParamRegistry":
        from tools.web.params import ParamRegistry as _P
        return _P
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DebugServer", "CameraManager", "ParamRegistry"]
