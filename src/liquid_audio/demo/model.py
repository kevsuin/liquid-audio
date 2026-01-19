"""Initialize models"""

import logging
import os
from dotenv import load_dotenv

import torch

from liquid_audio import LFM2AudioModel, LFM2AudioProcessor

load_dotenv()

logger = logging.getLogger(__name__)

__all__ = ["lfm2_audio", "mimi", "proc"]

HF_DIR = "LiquidAI/LFM2.5-Audio-1.5B"

# Device configuration: auto, mps, cuda, cpu
device_setting = os.getenv("LIQUID_AUDIO_DEVICE", "auto").lower()

if device_setting == "cpu":
    device = "cpu"
elif device_setting == "mps":
    device = "mps"
elif device_setting == "cuda":
    device = "cuda"
elif device_setting == "auto":
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
else:
    logger.warning(f"Unknown device setting: {device_setting}, using auto")
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

logger.info(f"Loading processor on device: {device}")
proc = LFM2AudioProcessor.from_pretrained(HF_DIR, device=device).eval()
logger.info("Loading model")
lfm2_audio = LFM2AudioModel.from_pretrained(HF_DIR, device=device).eval()
logger.info("Loading tokenizer")
mimi = proc.mimi.eval()

logger.info("Warmup tokenizer")
with mimi.streaming(1), torch.no_grad():
    for _ in range(5):
        x = torch.randint(2048, (1, 8, 1), device=proc.device)
        mimi.decode(x)
