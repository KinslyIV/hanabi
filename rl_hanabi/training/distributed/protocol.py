"""
Shared protocol definitions for GPU client-server communication.
Both gpu_server.py and gpu_client.py should import from here.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict


class TrainingRequest(BaseModel):
    """Request from coordinator to perform a training step."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    request_type: str  # 'train_step', 'get_weights', 'set_weights', 'ping', 'save_checkpoint', 'load_checkpoint', 'auth'
    payload: Optional[Dict[str, Any]] = None


class TrainingResponse(BaseModel):
    """Response to coordinator after completing request."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    success: bool
    response_type: str
    payload: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
