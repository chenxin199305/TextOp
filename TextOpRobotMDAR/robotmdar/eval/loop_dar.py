"""
Loop DAR Script - Continuous Motion Generation

Continuously generates motion using DAR model in an autoregressive manner.
Starts from zero pose and generates infinite trajectory based on text prompts.

Usage:
- python eval/loop_dar.py --config-name=loop_dar
- Interactive commands:
  - Input text in terminal: Change text prompt
  - Space or 'p': Pause/resume generation
  - Esc or 'q': Quit
"""

import threading
import time
import numpy as np
import sys

import clip
import numpy as np
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

from robotmdar.dtype import seed, logger as dtype_logger
from robotmdar.dtype.abc import Dataset, VAE, Denoiser, Diffusion, SSampler
from robotmdar.dtype.motion import (motion_dict_to_qpos, get_zero_abs_pose,
                                    motion_dict_to_abs_pose, get_zero_feature,
                                    FeatureVersion)
from robotmdar.dtype.vis_mjc import mjc_load_everything
from robotmdar.eval.generate_dar import ClassifierFreeWrapper, generate_next_motion
from robotmdar.train.manager import DARManager

# import torch_tensorrt
from robotmdar.wrapper.vae_decode import DecoderWrapper
from robotmdar.dtype.debug import pdb_decorator


# import torch_tensorrt


class LoopState:
    """State management for continuous motion generation."""

    def __init__(self):
        self.paused = False
        self.text_prompt = "stand"
        self.text_changed = True  # Start with True to encode initial text
        self.quit_requested = False


def get_text_embedding(text: str, clip_model, device: str) -> torch.Tensor:
    """Encode text using CLIP model."""
    try:
        with torch.no_grad():
            text_tokens = clip.tokenize([text]).to(device)
            text_embedding = clip_model.encode_text(text_tokens)
            # text_embedding = text_embedding / text_embedding.norm(dim=-1,
            #                                                       keepdim=True)
        return text_embedding.float()
    except Exception as e:
        logger.warning(f"Failed to encode text '{text}': {e}")
        return torch.zeros(1, 512, device=device, dtype=torch.float32)


def interactive_input_thread(loop_state: LoopState):
    """Interactive input thread for user commands."""
    while not loop_state.quit_requested:
        try:
            user_input = input()
            print(f"You entered new prompt: {user_input}")
            loop_state.text_prompt = user_input
            loop_state.text_changed = True
        except (EOFError, KeyboardInterrupt):
            break


def warmup(vae_trt, cfg_denoiser, diffusion, val_data, clip_model,
           history_motion, abs_pose, future_len, history_len, cfg):
    logger.info("Warming up...")
    text_embedding = get_text_embedding("stand", clip_model, cfg.device)
    future_motion, motion_dict, abs_pose = generate_next_motion(
        vae=vae_trt,
        denoiser=cfg_denoiser,
        diffusion=diffusion,
        val_data=val_data,
        text_embedding=text_embedding,
        history_motion=history_motion,
        abs_pose=abs_pose,
        future_len=future_len,
        # cfg=cfg,
        use_full_sample=cfg.use_full_sample,
        guidance_scale=cfg.guidance_scale,
    )
    # 因为第一次和第二次, history_motion的stride内存布局不一样, 会重新触发编译

    # print(f"First history_motion strides: {history_motion.stride()}")
    # print(
    #     f"Second history_motion strides: {future_motion[:, -history_len:, :].stride()}"
    # )
    history_motion = future_motion[:, -history_len:, :]
    future_motion, motion_dict, abs_pose = generate_next_motion(
        vae=vae_trt,
        denoiser=cfg_denoiser,
        diffusion=diffusion,
        val_data=val_data,
        text_embedding=text_embedding,
        history_motion=history_motion,
        abs_pose=abs_pose,
        future_len=future_len,
        # cfg=cfg,
        use_full_sample=cfg.use_full_sample,
        guidance_scale=cfg.guidance_scale,
    )
    logger.info("Warming up done")


@pdb_decorator
def main(cfg: DictConfig):
    dtype_logger.set(cfg)
    seed.set(cfg.seed)
    # torch.set_default_device(cfg.device)

    # Load models
    clip_model, _ = clip.load("ViT-B/32", device=cfg.device)
    clip_model.eval()

    val_data: Dataset = instantiate(cfg.data.val)
    vae: VAE = instantiate(cfg.vae)
    denoiser: Denoiser = instantiate(cfg.denoiser)

    schedule_sampler: SSampler = instantiate(cfg.diffusion.schedule_sampler)
    diffusion: Diffusion = schedule_sampler.diffusion

    vae.eval()
    denoiser.eval()

    # Load checkpoints
    manager: DARManager = instantiate(cfg.train.manager)
    manager.hold_model(vae, denoiser, None, val_data)

    # vae_trt = torch.compile(vae, backend='tensorrt')
    # denoiser_trt = torch.compile(denoiser, backend='tensorrt')
    vae_trt = vae
    denoiser_trt = denoiser
    cfg_denoiser = ClassifierFreeWrapper(denoiser_trt)

    future_len = cfg.data.future_len
    history_len = cfg.data.history_len

    # Initialize state
    loop_state = LoopState()

    # Initialize motion generation state
    if FeatureVersion == 4:
        init_motion = get_zero_feature(val_data.skeleton)
        history_motion = val_data.normalize(
            init_motion.unsqueeze(0).expand(1, history_len, -1).to(cfg.device))
    else:
        history_motion = val_data.normalize(
            get_zero_feature().unsqueeze(0).expand(1, history_len,
                                                   -1).to(cfg.device))
    abs_pose = get_zero_abs_pose((1,), device=cfg.device)
    text_embedding = None
    # warmup(vae_trt, cfg_denoiser, diffusion, val_data, clip_model,
    #        history_motion, abs_pose, future_len, history_len, cfg)

    # Setup visualization with keyboard callback
    dt = 1.0 / val_data.fps

    def keycb_fn(key):
        """Handle keyboard input for interactive control."""
        # Space (32) or 'p' key: pause/resume
        if key == ord(' ') or key == ord('P') or key == ord('p'):
            loop_state.paused = not loop_state.paused
            status = "paused" if loop_state.paused else "resumed"
            logger.info(f"Generation {status}")
        # Esc (256 is GLFW ESC, 27 is ASCII ESC) or 'q' key: quit
        elif key == 256 or key == 27 or key == ord('Q') or key == ord('q'):
            logger.info("Quit requested")
            loop_state.quit_requested = True

    show_fn, viewer = mjc_load_everything(dt, keycb_fn)

    # Start interactive input thread
    input_thread = threading.Thread(target=interactive_input_thread,
                                    args=(loop_state,))
    input_thread.daemon = True
    input_thread.start()

    logger.info("Starting continuous motion generation...")
    logger.info(
        "Commands: Input text in terminal (change prompt), Space/p(pause), Esc/q(quit)"
    )
    logger.info(f"Initial text: {loop_state.text_prompt}")

    # Main generation loop
    frame_idx = 0
    while not loop_state.quit_requested and viewer.is_running():
        # Update text embedding if changed
        if loop_state.text_changed:
            text_embedding = get_text_embedding(loop_state.text_prompt,
                                                clip_model, cfg.device)
            loop_state.text_changed = False
            logger.info(
                f"Updated text embedding for: {loop_state.text_prompt}")

        # Generate next motion if not paused
        if not loop_state.paused and text_embedding is not None:
            # breakpoint()
            future_motion, motion_dict, abs_pose = generate_next_motion(
                vae=vae_trt,
                denoiser=cfg_denoiser,
                diffusion=diffusion,
                val_data=val_data,
                text_embedding=text_embedding,
                history_motion=history_motion,
                abs_pose=abs_pose,
                future_len=future_len,
                # cfg=cfg,
                use_full_sample=cfg.use_full_sample,
                guidance_scale=cfg.guidance_scale)

            # Update history for next generation (autoregressive)
            history_motion = future_motion[:, -history_len:, :]

            # Visualize the motion
            qpos_data, contact_data = motion_dict_to_qpos(motion_dict)

            # Convert to numpy - qpos_data and contact_data are torch tensors
            qpos_np = qpos_data.detach().cpu().numpy()  # [B, T, 30]
            contact_np = contact_data.detach().cpu().numpy()  # [B, T, 2]

            # Show each frame of the generated motion
            for t in range(qpos_np.shape[1]):
                if loop_state.quit_requested or not viewer.is_running():
                    break
                show_fn(qpos_np[0, t], contact_np[0, t])
                time.sleep(dt)
                frame_idx += 1
                # print("Frame ID: ", frame_idx)
        else:
            time.sleep(0.1)  # Small sleep when paused or no text embedding

    logger.info("Shutting down...")
    viewer.close()
