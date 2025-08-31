import os
import torch
import random
import numpy as np
import folder_paths
import torchaudio
import gc
import logging
from PIL import Image

# Use relative imports for our vendored code
from .src.hunyuanvideo_foley.utils.model_utils import denoise_process
from .src.hunyuanvideo_foley.utils.config_utils import load_yaml, AttributeDict
from .src.hunyuanvideo_foley.utils.feature_utils import encode_video_with_sync, encode_text_feat
from .src.hunyuanvideo_foley.constants import FPS_VISUAL
from .src.hunyuanvideo_foley.models.dac_vae.model.dac import DAC

# --- Helper Functions ---
logging.basicConfig(level=logging.INFO, format='HunyuanFoley (%(levelname)s): %(message)s')

def set_manual_seed(seed):
    seed = int(seed)
    numpy_seed = seed % (2**32)
    random.seed(seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def empty_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def load_state_dict(model, model_path):
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys: logging.warning(f"Missing keys: {missing_keys}")
    if unexpected_keys: logging.warning(f"Unexpected keys: {unexpected_keys}")
    return model

# --- Model Cache ---
loaded_models_cache = {}
loaded_vaes_cpu = {}

def unload_foley_models():
    global loaded_models_cache
    if not loaded_models_cache:
        logging.info("No Hunyuan-Foley models to unload.")
        return

    logging.info("Unloading Hunyuan-Foley models from VRAM to CPU as requested.")
    try:
        for key in loaded_models_cache:
            model_tuple = loaded_models_cache[key]
            model_dict = model_tuple[0]
            for model_name, model in model_dict.items():
                if hasattr(model, 'to'):
                    model_dict[model_name] = model.to("cpu")
        
        empty_cuda_cache()
        logging.info("Hunyuan-Foley models successfully moved to CPU and VRAM cleared.")
    except Exception as e:
        logging.error(f"An error occurred while unloading models: {e}")


class HunyuanFoleyModelLoader:
    def __init__(self):
        self.model_dir = os.path.join(folder_paths.models_dir, "hunyuan_foley")

    @classmethod
    def INPUT_TYPES(s):
        instance = s()
        if not os.path.exists(instance.model_dir): os.makedirs(instance.model_dir)
        
        model_paths = []
        root_folder_name = os.path.basename(instance.model_dir)

        if os.path.isfile(os.path.join(instance.model_dir, "hunyuanvideo_foley.pth")):
            model_paths.append(root_folder_name)
        
        for f in os.listdir(instance.model_dir):
            sub_path = os.path.join(instance.model_dir, f)
            if os.path.isdir(sub_path) and os.path.isfile(os.path.join(sub_path, "hunyuanvideo_foley.pth")):
                model_paths.append(f)
        
        if not model_paths:
            return {"required": {"error": ("STRING", {"default": "No models found. See console for instructions on folder structure.", "multiline": True})}}
            
        return {"required": {"model_path_name": (model_paths,)}}

    RETURN_TYPES = ("FOLEY_MODEL",)
    FUNCTION = "load_foley_model"
    CATEGORY = "HunyuanVideo-Foley"

    def load_foley_model(self, model_path_name, error=None):
        if error:
            logging.error("No model folders found in ComfyUI/models/hunyuan_foley/.")
            raise ValueError(error)

        global loaded_models_cache
        
        root_folder_name = os.path.basename(self.model_dir)
        if model_path_name == root_folder_name:
            model_path = self.model_dir
        else:
            model_path = os.path.join(self.model_dir, model_path_name)

        config_name = "hunyuanvideo-foley-xxl.yaml"
        base_dir = os.path.dirname(__file__)
        config_path = os.path.join(base_dir, "src/hunyuanvideo_foley/configs", config_name)
        
        precision = "bfloat16"
        cache_key = (os.path.normpath(model_path), precision)
        
        if cache_key in loaded_models_cache:
            logging.info(f"Loading cached models from {model_path_name}")
            return (loaded_models_cache[cache_key],)

        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Loading all models to {target_device} for the first time. This may take a moment...")
        model_dict, cfg = self.load_all_models_to_vram(model_path, config_path, precision, target_device)
        
        foley_model_tuple = (model_dict, cfg, precision)
        loaded_models_cache[cache_key] = foley_model_tuple
        return (foley_model_tuple,)

    def load_all_models_to_vram(self, model_path, config_path, precision, target_device):
        from .src.hunyuanvideo_foley.models.hifi_foley import HunyuanVideoFoley
        from .src.hunyuanvideo_foley.models.synchformer import Synchformer
        from transformers import AutoTokenizer, ClapTextModelWithProjection, SiglipImageProcessor, SiglipVisionModel
        
        cfg = load_yaml(config_path)
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[precision]

        siglip_path = os.path.join(model_path, "siglip2")
        clap_path = os.path.join(model_path, "clap")
        
        if not os.path.isdir(siglip_path): raise FileNotFoundError(f"SigLIP2 folder not found at {siglip_path}")
        if not os.path.isdir(clap_path): raise FileNotFoundError(f"CLAP folder not found at {clap_path}")

        foley_model = HunyuanVideoFoley(cfg, dtype=dtype).eval()
        load_state_dict(foley_model, os.path.join(model_path, "hunyuanvideo_foley.pth"))
        
        siglip2_model = SiglipVisionModel.from_pretrained(siglip_path, local_files_only=True, low_cpu_mem_usage=True).eval()
        siglip2_preprocess = SiglipImageProcessor.from_pretrained(siglip_path, local_files_only=True)

        clap_tokenizer = AutoTokenizer.from_pretrained(clap_path, local_files_only=True)
        clap_model = ClapTextModelWithProjection.from_pretrained(clap_path, local_files_only=True, low_cpu_mem_usage=True).eval()
        
        from torchvision.transforms import v2
        syncformer_preprocess = v2.Compose([v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC), v2.CenterCrop(224), v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        syncformer_model = Synchformer().eval()
        syncformer_model.load_state_dict(torch.load(os.path.join(model_path, "synchformer_state_dict.pth"), map_location="cpu"))
        
        model_dict = { 
            'foley_model': foley_model.to(target_device, dtype=dtype), 
            'siglip2_preprocess': siglip2_preprocess, 'siglip2_model': siglip2_model.to(target_device), 
            'clap_tokenizer': clap_tokenizer, 'clap_model': clap_model.to(target_device), 
            'syncformer_preprocess': syncformer_preprocess, 'syncformer_model': syncformer_model.to(target_device)
        }
        return model_dict, cfg


class LoadDACHunyuanVAE:
    @classmethod
    def INPUT_TYPES(s):
        vae_files = [f for f in folder_paths.get_filename_list("vae") if "dac" in f.lower() or "vae_128d_48k" in f.lower()]
        model_dir = os.path.join(folder_paths.models_dir, "hunyuan_foley")
        if os.path.exists(model_dir):
            for root, _, files in os.walk(model_dir):
                for file in files:
                    if file == "vae_128d_48k.pth":
                        rel_path = os.path.relpath(os.path.join(root, file), folder_paths.models_dir)
                        if rel_path not in vae_files: vae_files.append(rel_path)
        return {"required": {"vae_name": (vae_files,)}}

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "HunyuanVideo-Foley"

    def load_vae(self, vae_name):
        global loaded_vaes_cpu
        vae_path = folder_paths.get_full_path("vae", vae_name)
        if not vae_path: vae_path = os.path.join(folder_paths.models_dir, vae_name)
        if vae_path in loaded_vaes_cpu: return (loaded_vaes_cpu[vae_path],)
        vae = DAC.load(vae_path).eval()
        loaded_vaes_cpu[vae_path] = vae
        return (vae,)


class HunyuanFoleySampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "foley_model": ("FOLEY_MODEL",),
            "video_frames": ("IMAGE",),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 1.0}),
            "prompt": ("STRING", {"multiline": True, "default": "A person walks on frozen ice"}),
            # --- NEW ---: Added negative_prompt input
            "negative_prompt": ("STRING", {"multiline": True, "default": "noisy, harsh"}),
            "guidance_scale": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 10.0, "step": 0.1}),
            "steps": ("INT", {"default": 50, "min": 10, "max": 100, "step": 1}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "HunyuanVideo-Foley"

    # --- MODIFIED ---: Added negative_prompt to the function signature
    def sample(self, foley_model, video_frames, fps, prompt, negative_prompt, guidance_scale, steps, seed):
        model_dict, cfg, precision = foley_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        current_device_type = next(model_dict['foley_model'].parameters()).device.type
        if current_device_type != device:
            logging.info(f"Models are on '{current_device_type}', moving to '{device}' for sampling...")
            dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[precision]
            for model_name, model in model_dict.items():
                if hasattr(model, 'to'):
                    if model_name == 'foley_model':
                        model_dict[model_name] = model.to(device, dtype=dtype)
                    else:
                        model_dict[model_name] = model.to(device)
            empty_cuda_cache()
            logging.info("Models successfully moved for sampling.")
        
        set_manual_seed(seed)
        
        frames_np = (video_frames.cpu().numpy() * 255).astype(np.uint8)
        all_frames = [frame for frame in frames_np]
        num_frames = len(all_frames)
        
        audio_len_in_s = num_frames / fps
        logging.info(f"Received {num_frames} frames at {fps} FPS, for a total duration of {audio_len_in_s:.2f}s.")

        siglip_fps = FPS_VISUAL["siglip2"]
        siglip_indices = np.linspace(0, num_frames - 1, int(audio_len_in_s * siglip_fps)).astype(int)
        frames_siglip = [all_frames[i] for i in siglip_indices]

        images_siglip = model_dict['siglip2_preprocess'](images=[Image.fromarray(f).convert('RGB') for f in frames_siglip], return_tensors="pt").to(device)
        with torch.no_grad():
            siglip_output = model_dict['siglip2_model'](**images_siglip)
        siglip_feat = siglip_output.pooler_output.unsqueeze(0)

        sync_fps = FPS_VISUAL["synchformer"]
        sync_indices = np.linspace(0, num_frames - 1, int(audio_len_in_s * sync_fps)).astype(int)
        frames_sync_np = np.array([all_frames[i] for i in sync_indices])
        
        images_sync = torch.from_numpy(frames_sync_np).permute(0, 3, 1, 2)
        sync_frames = model_dict['syncformer_preprocess'](images_sync).unsqueeze(0)
        
        model_dict_with_device = {**model_dict, 'device': device}
        sync_feat = encode_video_with_sync(sync_frames, AttributeDict(model_dict_with_device))

        # --- MODIFIED ---: Replaced the hardcoded string with the new input variable
        prompts = [negative_prompt, prompt]
        text_feat_res, _ = encode_text_feat(prompts, AttributeDict(model_dict_with_device))
        text_feat, uncond_text_feat = text_feat_res[1:], text_feat_res[:1]
        if cfg.model_config.model_kwargs.text_length < text_feat.shape[1]:
            text_feat, uncond_text_feat = text_feat[:, :cfg.model_config.model_kwargs.text_length], uncond_text_feat[:, :cfg.model_config.model_kwargs.text_length]
        
        logging.info(f"Generating latents for {audio_len_in_s:.2f}s of audio...")
        
        latents = denoise_process(
            AttributeDict({'siglip2_feat': siglip_feat, 'syncformer_feat': sync_feat}), 
            AttributeDict({'text_feat': text_feat, 'uncond_text_feat': uncond_text_feat}), 
            audio_len_in_s, 
            AttributeDict({'foley_model': model_dict['foley_model'], 'device': device}), 
            cfg, guidance_scale, steps
        )
        
        return ({"samples": latents.cpu(), "audio_len_in_s": audio_len_in_s},)


class DACHunyuanVAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "samples": ("LATENT",), 
                "vae": ("VAE",) 
            },
            "optional": {
                "unload_models_after_use": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "decode"
    CATEGORY = "HunyuanVideo-Foley"

    def decode(self, samples, vae, unload_models_after_use=False):
        latents = samples["samples"]
        audio_len_in_s = samples["audio_len_in_s"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logging.info("Moving VAE to GPU for decoding...")
        vae_on_device = vae.to(device)
        try:
            with torch.no_grad():
                audio_tensor = vae_on_device.decode(latents.to(device)).float().cpu()
            sample_rate = vae.sample_rate
            audio_tensor = audio_tensor[:, :int(audio_len_in_s * sample_rate)]
            audio_out = {"waveform": audio_tensor, "sample_rate": sample_rate}
        finally:
            vae_on_device.to("cpu")
            empty_cuda_cache()
            logging.info("VAE decoding complete and moved to CPU.")

            if unload_models_after_use:
                unload_foley_models()

        return (audio_out,)

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "HunyuanFoleyModelLoader": HunyuanFoleyModelLoader,
    "LoadDACHunyuanVAE": LoadDACHunyuanVAE,
    "HunyuanFoleySampler": HunyuanFoleySampler,
    "DACHunyuanVAEDecode": DACHunyuanVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanFoleyModelLoader": "Hunyuan-Foley model loader",
    "LoadDACHunyuanVAE": "Hunyuan-Foley VAE loader",
    "HunyuanFoleySampler": "Hunyuan-Foley Sampler",
    "DACHunyuanVAEDecode": "Hunyuan-Foley VAE Decode",
}