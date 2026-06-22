import time
import abc
from typing import Any, Dict
from pydantic import BaseModel

class ToolResult(BaseModel):
    """Standardized output structure for all tools to keep state updates deterministic."""
    success: bool
    result_data: Any
    error_message: str | None = None
    execution_time_ms: int

class BaseTool(abc.ABC):
    """
    Abstract Base Class that every tool must implement.
    Enforces standardized telemetry tracking (latency, errors) required by our state machine.
    """
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abc.abstractmethod
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Internal execution logic to be overridden by child tools."""
        pass

    def run(self, *args: Any, **kwargs: Any) -> ToolResult:
        """Wrapper execution block that automatically captures errors and performance telemetry."""
        start_time = time.time()
        try:
            data = self._run(*args, **kwargs)
            duration = int((time.time() - start_time) * 1000)
            return ToolResult(success=True, result_data=data, execution_time_ms=duration)
        except Exception as e:
            duration = int((time.time() - start_time) * 1000)
            return ToolResult(
                success=False, 
                result_data=None, 
                error_message=f"Error in tool '{self.name}': {str(e)}", 
                execution_time_ms=duration
            )