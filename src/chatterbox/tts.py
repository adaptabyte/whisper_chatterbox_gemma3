import torch

class ChatterboxTTS:
    """Stub replacement for the missing chatterbox TTS implementation."""

    def __init__(self, device="cpu"):
        self.device = device
        self.sr = 22050

    @classmethod
    def from_pretrained(cls, device="cpu"):
        return cls(device)

    def to(self, device):
        self.device = device

    def generate_stream(self, text, audio_prompt_path=None, chunk_size=25, **kwargs):
        # This stub simply returns silence as a placeholder audio output.
        duration = 1.0  # seconds of silence
        samples = int(self.sr * duration)
        tensor = torch.zeros(1, samples, dtype=torch.float32)
        yield tensor, {}
