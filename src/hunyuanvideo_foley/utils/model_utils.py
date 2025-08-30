import torch
from diffusers.utils.torch_utils import randn_tensor
from .schedulers import FlowMatchDiscreteScheduler
from tqdm import tqdm

def retrieve_timesteps(scheduler, num_inference_steps, device, **kwargs):
    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    return scheduler.timesteps, num_inference_steps

def prepare_latents(scheduler, batch_size, num_channels_latents, length, dtype, device):
    shape = (batch_size, num_channels_latents, int(length))
    latents = randn_tensor(shape, device=device, dtype=dtype)
    if hasattr(scheduler, "init_noise_sigma"):
        latents = latents * scheduler.init_noise_sigma
    return latents

@torch.no_grad()
def denoise_process(visual_feats, text_feats, audio_len_in_s, model_dict, cfg, guidance_scale=4.5, num_inference_steps=50, batch_size=1):
    device = model_dict.device
    foley_model = model_dict.foley_model
    target_dtype = foley_model.dtype
    autocast_enabled = target_dtype != torch.float32
    
    scheduler = FlowMatchDiscreteScheduler(
        shift=cfg.diffusion_config.sample_flow_shift, reverse=cfg.diffusion_config.flow_reverse,
        solver=cfg.diffusion_config.flow_solver, use_flux_shift=cfg.diffusion_config.sample_use_flux_shift,
        flux_base_shift=cfg.diffusion_config.flux_base_shift, flux_max_shift=cfg.diffusion_config.flux_max_shift)
    
    timesteps, _ = retrieve_timesteps(scheduler, num_inference_steps, device)
    
    latents = prepare_latents(
        scheduler, batch_size, cfg.model_config.model_kwargs.audio_vae_latent_dim,
        audio_len_in_s * cfg.model_config.model_kwargs.audio_frame_rate, target_dtype, device)

    progress_bar = tqdm(timesteps, desc="Denoising steps")
    for t in progress_bar:
        latent_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
        latent_input = scheduler.scale_model_input(latent_input, t)
        t_expand = t.repeat(latent_input.shape[0])

        siglip2_feat = visual_feats.siglip2_feat.to(device)
        uncond_siglip2_feat = foley_model.get_empty_clip_sequence(bs=batch_size, len=siglip2_feat.shape[1]).to(device)
        siglip2_feat_input = torch.cat([uncond_siglip2_feat, siglip2_feat]) if guidance_scale > 1.0 else siglip2_feat

        syncformer_feat = visual_feats.syncformer_feat.to(device)
        uncond_syncformer_feat = foley_model.get_empty_sync_sequence(bs=batch_size, len=syncformer_feat.shape[1]).to(device)
        syncformer_feat_input = torch.cat([uncond_syncformer_feat, syncformer_feat]) if guidance_scale > 1.0 else syncformer_feat

        text_feat = text_feats.text_feat.to(device)
        uncond_text_feat = text_feats.uncond_text_feat.to(device)
        text_feat_input = torch.cat([uncond_text_feat, text_feat]) if guidance_scale > 1.0 else text_feat

        with torch.autocast(device_type=device, enabled=autocast_enabled, dtype=target_dtype):
            noise_pred = foley_model(
                x=latent_input, t=t_expand, cond=text_feat_input,
                clip_feat=siglip2_feat_input, sync_feat=syncformer_feat_input, return_dict=True)["x"]

        noise_pred = noise_pred.to(torch.float32)
        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    return latents