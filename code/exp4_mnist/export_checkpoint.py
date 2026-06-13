"""Export the pretrained DDPM checkpoint for the standalone Colab notebooks.

Saves a single .pt file with the UNet config + weights and the scheduler
config.  Upload it next to the notebook in Colab (they expect the file
name `ddpm_mnist_pretrained.pt`).

Usage:
    python export_checkpoint.py [output.pt]
"""

import sys
from pathlib import Path

import torch
from diffusers import DDPMPipeline

MODEL_ID = '1aurent/ddpm-mnist'


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('ddpm_mnist_pretrained.pt')

    pipe = DDPMPipeline.from_pretrained(MODEL_ID)
    checkpoint = {
        'unet_config': dict(pipe.unet.config),
        'unet_state_dict': pipe.unet.state_dict(),
        'scheduler_config': dict(pipe.scheduler.config),
    }
    torch.save(checkpoint, out)
    print(f'Saved {MODEL_ID} -> {out}')


if __name__ == '__main__':
    main()
