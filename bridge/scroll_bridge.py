# scroll_bridge.py

from scrolls.scroll_engine import cast_scroll
from soul.tenets import TENETS
import logging

class ScrollBridge:
    def __init__(self, codex=None, runtime=None):
        self.codex = codex
        self.runtime = runtime
        self.logger = logging.getLogger("ScrollBridge")

    def invoke_scroll(self, scroll_name, **kwargs):
        self.logger.info(f"Invoking scroll: {scroll_name} with args: {kwargs}")
        try:
            result = cast_scroll(scroll_name, context=self.codex, runtime=self.runtime, **kwargs)
            return result
        except Exception as e:
            self.logger.error(f"Scroll invocation failed: {e}")
            return {"status": "error", "message": str(e)}

    def list_available_scrolls(self):
        return list(self.codex.get("scroll_registry", {}).keys()) if self.codex else []

    def verify_tenet_compatibility(self, scroll_name):
        if scroll_name in TENETS.get("restricted_scrolls", []):
            return False
        return True
