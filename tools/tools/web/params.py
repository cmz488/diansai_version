"""Parameter registry with type validation and change callbacks."""

import threading
from typing import Any, Callable, Dict, List, Optional


class ParamRegistry:
    """Registry of tunable parameters with validation and callbacks.

    Parameters are shared across all connected browser clients.
    Changes trigger registered callbacks (e.g., WebSocket broadcast).
    """

    VALID_TYPES = ("int", "float", "bool", "choice")

    def __init__(self):
        self._params: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[str, Any], None]] = []

    # ── registration ──────────────────────────────────────────

    def add(
        self,
        name: str,
        type: str = "int",
        default: Any = 0,
        range: Optional[tuple] = None,
        step: Any = None,
        choices: Optional[list] = None,
        group: str = "默认",
        description: str = "",
    ) -> Dict:
        """Register a parameter. Returns the param definition dict."""
        # Normalize: accept both Python types (int, float, bool) and strings ("int", "float", "bool", "choice")
        if not isinstance(type, str):
            type = type.__name__
        if type not in self.VALID_TYPES:
            raise ValueError(f"Invalid type '{type}', must be one of {self.VALID_TYPES}")
        if type == "choice" and not choices:
            raise ValueError("'choice' type requires a 'choices' list")

        with self._lock:
            self._params[name] = {
                "name": name,
                "type": type,
                "value": default,
                "default": default,
                "range": range,
                "step": step,
                "choices": choices,
                "group": group,
                "description": description,
            }
        return self._params[name]

    # ── read / write ──────────────────────────────────────────

    def get(self, name: str) -> Any:
        """Return the current value of a parameter."""
        with self._lock:
            if name not in self._params:
                raise KeyError(f"Unknown parameter: {name}")
            return self._params[name]["value"]

    def set(self, name: str, value: Any) -> bool:
        """Validate and update a parameter. Returns True on success."""
        with self._lock:
            if name not in self._params:
                return False
            p = self._params[name]
            if not self._validate(p, value):
                return False
            old = p["value"]
            p["value"] = value
        if value != old:
            for cb in self._callbacks:
                try:
                    cb(name, value)
                except Exception:
                    pass
        return True

    def snapshot(self) -> Dict[str, Any]:
        """Return {name: current_value} for all params."""
        with self._lock:
            return {name: p["value"] for name, p in self._params.items()}

    def list_all(self) -> List[Dict]:
        """Return full definitions of all params (for frontend init)."""
        with self._lock:
            return [dict(p) for p in self._params.values()]

    def on_change(self, callback: Callable[[str, Any], None]):
        """Register a callback invoked on every successful param change."""
        self._callbacks.append(callback)

    # ── internals ─────────────────────────────────────────────

    def _validate(self, p: Dict, value: Any) -> bool:
        """Type-check and range-check a single value."""
        t = p["type"]
        try:
            if t == "int":
                value = int(value)
            elif t == "float":
                value = float(value)
            elif t == "bool":
                value = bool(value)
            elif t == "choice":
                if value not in p["choices"]:
                    return False
        except (ValueError, TypeError):
            return False

        if t in ("int", "float") and p["range"] is not None:
            lo, hi = p["range"]
            if not (lo <= value <= hi):
                return False

        return True
