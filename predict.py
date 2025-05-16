from cog import BasePredictor, Input, Path
import torch
import torchaudio
import os
import requests
from huggingface_hub import hf_hub_download
from indextts.infer import IndexTTS

MODEL_FILES = {
    "config.yaml": "config.yaml",
    "bigvgan_discriminator.pth": "bigvgan_discriminator.pth",
    "bigvgan_generator.pth": "bigvgan_generator.pth",
    "bpe.model": "bpe.model",
    "dvae.pth": "dvae.pth",
    "gpt.pth": "gpt.pth",
    "unigram_12000.vocab": "unigram_12000.vocab"
}

MODEL_REPO = "IndexTeam/IndexTTS-1.5"
CHECKPOINTS_DIR = "checkpoints"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # Create checkpoints directory if it doesn't exist
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

        # Download all required model files
        for filename in MODEL_FILES.values():
            output_path = os.path.join(CHECKPOINTS_DIR, filename)
            if not os.path.exists(output_path):
                print(f"Downloading {filename}...")
                try:
                    hf_hub_download(
                        repo_id=MODEL_REPO,
                        filename=filename,
                        local_dir=CHECKPOINTS_DIR,
                        local_dir_use_symlinks=False
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to download {filename}: {str(e)}")

        # Initialize model
        try:
            self.model = IndexTTS(
                cfg_path=os.path.join(CHECKPOINTS_DIR, "config.yaml"),
                model_dir=CHECKPOINTS_DIR,
                is_fp16=True,
                use_cuda_kernel=False
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def predict(
        self,
        text: str = Input(description="Text to synthesize"),
        reference_audio: Path = Input(description="Reference audio file for voice cloning"),
    ) -> Path:
        """Run a single prediction on the model"""
        try:
            # Process inputs
            if not os.path.exists(reference_audio):
                raise ValueError("Reference audio file not found")

            # Generate audio
            output_path = "output.wav"
            self.model.infer(
                audio_prompt=str(reference_audio),
                text=text,
                output_path=output_path
            )

            return Path(output_path)

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")