import os
import torch
import random
import numpy as np
import folder_paths
import torchaudio
import gc
import logging
import sys
import contextlib
import warnings
from PIL import Image

# --- Suppress noisy logs and warnings ---
warnings.filterwarnings("ignore")

# Force diffusers and transformers to be quiet as early as possible
try:
    import transformers
    transformers.utils.logging.set_verbosity_error()
except ImportError:
    pass

try:
    import diffusers.utils.logging
    diffusers.utils.logging.set_verbosity_error()
except ImportError:
    pass

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

class DummyFile:
    def write(self, x): pass
    def flush(self): pass
    def isatty(self): return False

@contextlib.contextmanager
def suppress_output():
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = DummyFile()
    sys.stderr = DummyFile()
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

# --- Workaround for diffusers bug ---
# Fix for: NameError: name 'logger' is not defined in diffusers.quantizers.torchao.torchao_quantizer
# We do this WITHOUT suppression first to avoid capturing a closed file in the logger
try:
    import diffusers.utils.logging
    import builtins
    if not hasattr(builtins, 'logger'):
        # Inject a logger that is already set to ERROR
        l = diffusers.utils.logging.get_logger("diffusers.quantizers.torchao.torchao_quantizer")
        l.setLevel(logging.ERROR)
        builtins.logger = l
except Exception:
    pass

from comfy_api.latest import ComfyExtension, io, ui

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
    model.load_state_dict(state_dict, strict=False)
    return model

# --- Model Cache ---
loaded_models_cache = {}
loaded_vaes_cpu = {}

def unload_foley_models():
    global loaded_models_cache
    if not loaded_models_cache:
        return

    logging.info("Unloading Hunyuan-Foley models from VRAM to CPU...")
    try:
        for key in loaded_models_cache:
            model_tuple = loaded_models_cache[key]
            model_dict = model_tuple[0]
            for model_name, model in model_dict.items():
                if hasattr(model, 'to'):
                    model_dict[model_name] = model.to("cpu")
        
        empty_cuda_cache()
    except Exception as e:
        logging.error(f"An error occurred while unloading models: {e}")


class HunyuanFoleyModelLoader(io.ComfyNode):
    def __init__(self):
        self.model_dir = os.path.join(folder_paths.models_dir, "hunyuan_foley")

    @classmethod
    def define_schema(cls) -> io.Schema:
        model_dir = os.path.join(folder_paths.models_dir, "hunyuan_foley")
        if not os.path.exists(model_dir): os.makedirs(model_dir)
        
        model_paths = []
        root_folder_name = os.path.basename(model_dir)

        checkpoints = ["hunyuanvideo_foley.pth", "hunyuanvideo_foley_xl.pth"]
        
        def has_checkpoint(path):
            return any(os.path.isfile(os.path.join(path, cp)) for cp in checkpoints)

        if has_checkpoint(model_dir):
            model_paths.append(root_folder_name)
        
        for f in os.listdir(model_dir):
            sub_path = os.path.join(model_dir, f)
            if os.path.isdir(sub_path) and has_checkpoint(sub_path):
                model_paths.append(f)
        
        if not model_paths:
            return io.Schema(
                node_id="HunyuanFoleyModelLoader",
                display_name="Hunyuan-Foley model loader",
                category="HunyuanVideo-Foley",
                inputs=[
                    io.String.Input("error", default="No models found. See console for instructions on folder structure.", multiline=True)
                ],
                outputs=[]
            )
            
        return io.Schema(
            node_id="HunyuanFoleyModelLoader",
            display_name="Hunyuan-Foley model loader",
            category="HunyuanVideo-Foley",
            inputs=[
                io.Combo.Input("model_path_name", options=model_paths),
                io.Combo.Input("foley_checkpoint_name", options=["hunyuanvideo_foley.pth", "hunyuanvideo_foley_xl.pth"])
            ],
            outputs=[
                io.Custom("FOLEY_MODEL").Output()
            ]
        )

    @classmethod
    def execute(cls, model_path_name, foley_checkpoint_name, error=None) -> io.NodeOutput:
        if error:
            logging.error("No model folders found in ComfyUI/models/hunyuan_foley/.")
            raise ValueError(error)

        global loaded_models_cache
        
        model_dir = os.path.join(folder_paths.models_dir, "hunyuan_foley")
        root_folder_name = os.path.basename(model_dir)
        if model_path_name == root_folder_name:
            model_path = model_dir
        else:
            model_path = os.path.join(model_dir, model_path_name)

        precision = "bfloat16"
        cache_key = (os.path.normpath(model_path), precision, foley_checkpoint_name)

        models_to_unload = []
        for key, (model_dict, _, _) in loaded_models_cache.items():
            if key != cache_key:
                main_model = model_dict.get('foley_model')
                if main_model is not None and main_model.device.type == 'cuda':
                    models_to_unload.append(key)
        
        if models_to_unload:
            logging.info(f"Switching models. Unloading {len(models_to_unload)} model(s) from VRAM to CPU...")
            for key in models_to_unload:
                model_dict, _, _ = loaded_models_cache[key]
                for model in model_dict.values():
                    if hasattr(model, 'to'):
                        model.to('cpu')
            empty_cuda_cache()

        if foley_checkpoint_name == "hunyuanvideo_foley_xl.pth":
            config_name = "hunyuanvideo-foley-xl.yaml"
        else:
            config_name = "hunyuanvideo-foley-xxl.yaml"
        
        base_dir = os.path.dirname(__file__)
        config_path = os.path.join(base_dir, "src/hunyuanvideo_foley/configs", config_name)
        
        if cache_key in loaded_models_cache:
            return io.NodeOutput(loaded_models_cache[cache_key])

        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Loading {foley_checkpoint_name}...")
        model_dict, cfg = cls.load_all_models_to_vram(model_path, config_path, precision, target_device, foley_checkpoint_name)
        
        foley_model_tuple = (model_dict, cfg, precision)
        loaded_models_cache[cache_key] = foley_model_tuple
        return io.NodeOutput(foley_model_tuple)

    @classmethod
    def load_all_models_to_vram(cls, model_path, config_path, precision, target_device, foley_checkpoint_name):
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
        with suppress_output():
            load_state_dict(foley_model, os.path.join(model_path, foley_checkpoint_name))
        
        with suppress_output():
            siglip2_model = SiglipVisionModel.from_pretrained(siglip_path, local_files_only=True, low_cpu_mem_usage=True).eval()
            siglip2_preprocess = SiglipImageProcessor.from_pretrained(siglip_path, local_files_only=True)

            clap_tokenizer = AutoTokenizer.from_pretrained(clap_path, local_files_only=True)
            clap_model = ClapTextModelWithProjection.from_pretrained(clap_path, local_files_only=True, low_cpu_mem_usage=True).eval()
        
        from torchvision.transforms import v2
        
        # Preprocessor for PIL/Numpy inputs (legacy/backup)
        syncformer_preprocess = v2.Compose([
            v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC), 
            v2.CenterCrop(224), 
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True), 
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Preprocessor for Tensor inputs (B, C, H, W) in [0, 1]
        syncformer_preprocess_tensor = v2.Compose([
            v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(224),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        syncformer_model = Synchformer().eval()
        with suppress_output():
            syncformer_model.load_state_dict(torch.load(os.path.join(model_path, "synchformer_state_dict.pth"), map_location="cpu"))
        
        model_dict = { 
            'foley_model': foley_model.to(target_device, dtype=dtype), 
            'siglip2_preprocess': siglip2_preprocess, 'siglip2_model': siglip2_model.to(target_device), 
            'clap_tokenizer': clap_tokenizer, 'clap_model': clap_model.to(target_device), 
            'syncformer_preprocess': syncformer_preprocess,
            'syncformer_preprocess_tensor': syncformer_preprocess_tensor,
            'syncformer_model': syncformer_model.to(target_device)
        }
        return model_dict, cfg


class LoadDACHunyuanVAE(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        vae_files = [f for f in folder_paths.get_filename_list("vae") if "dac" in f.lower() or "vae_128d_48k" in f.lower()]
        model_dir = os.path.join(folder_paths.models_dir, "hunyuan_foley")
        if os.path.exists(model_dir):
            for root, _, files in os.walk(model_dir):
                for file in files:
                    if file == "vae_128d_48k.pth":
                        rel_path = os.path.relpath(os.path.join(root, file), folder_paths.models_dir)
                        if rel_path not in vae_files: vae_files.append(rel_path)
        return io.Schema(
            node_id="LoadDACHunyuanVAE",
            display_name="Hunyuan-Foley VAE loader",
            category="HunyuanVideo-Foley",
            inputs=[
                io.Combo.Input("vae_name", options=vae_files)
            ],
            outputs=[
                io.Vae.Output()
            ]
        )

    @classmethod
    def execute(cls, vae_name) -> io.NodeOutput:
        global loaded_vaes_cpu
        vae_path = folder_paths.get_full_path("vae", vae_name)
        if not vae_path: vae_path = os.path.join(folder_paths.models_dir, vae_name)
        if vae_path in loaded_vaes_cpu: return io.NodeOutput(loaded_vaes_cpu[vae_path])
        with suppress_output():
            vae = DAC.load(vae_path).eval()
        loaded_vaes_cpu[vae_path] = vae
        return io.NodeOutput(vae)


class HunyuanFoleySampler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="HunyuanFoleySampler",
            display_name="Hunyuan-Foley Sampler",
            category="HunyuanVideo-Foley",
            inputs=[
                io.Custom("FOLEY_MODEL").Input("foley_model"),
                io.Image.Input("video_frames"),
                io.Float.Input("fps", default=24.0, min=1.0, max=240.0, step=1.0),
                io.String.Input("prompt", default="A person walks on frozen ice", multiline=True),
                io.String.Input("negative_prompt", default="noisy, harsh", multiline=True),
                io.Float.Input("guidance_scale", default=4.5, min=1.0, max=10.0, step=0.1),
                io.Int.Input("steps", default=50, min=10, max=100, step=1),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF)
            ],
            outputs=[
                io.Latent.Output()
            ]
        )

    @classmethod
    def execute(cls, foley_model, video_frames, fps, prompt, negative_prompt, guidance_scale, steps, seed) -> io.NodeOutput:
        model_dict, cfg, precision = foley_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        current_device_type = next(model_dict['foley_model'].parameters()).device.type
        if current_device_type != device:
            logging.info(f"Moving models to '{device}'...")
            dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[precision]
            for model_name, model in model_dict.items():
                if hasattr(model, 'to'):
                    if model_name == 'foley_model':
                        model_dict[model_name] = model.to(device, dtype=dtype)
                    else:
                        model_dict[model_name] = model.to(device)
            empty_cuda_cache()
        
        set_manual_seed(seed)
        
        num_frames = video_frames.shape[0]
        audio_len_in_s = num_frames / fps
        
        # SigLIP Processing
        siglip_fps = FPS_VISUAL["siglip2"]
        siglip_indices = np.linspace(0, num_frames - 1, int(audio_len_in_s * siglip_fps)).astype(int)
        
        # Convert selected frames to PIL for SigLIP (safest for HF processor)
        # Slicing tensor first avoids converting all frames
        frames_siglip_tensor = video_frames[siglip_indices] # (B_sub, H, W, C)
        frames_siglip_np = (frames_siglip_tensor.cpu().numpy() * 255).astype(np.uint8)
        images_siglip_pil = [Image.fromarray(f).convert('RGB') for f in frames_siglip_np]

        images_siglip = model_dict['siglip2_preprocess'](images=images_siglip_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            siglip_output = model_dict['siglip2_model'](**images_siglip)
        siglip_feat = siglip_output.pooler_output.unsqueeze(0)

        # Synchformer Processing
        sync_fps = FPS_VISUAL["synchformer"]
        sync_indices = np.linspace(0, num_frames - 1, int(audio_len_in_s * sync_fps)).astype(int)
        
        # Direct Tensor Processing for Synchformer
        # Input: (B, H, W, C) -> Subselect -> Permute (B_sub, C, H, W)
        frames_sync_tensor = video_frames[sync_indices].permute(0, 3, 1, 2)
        
        # Use optimized tensor preprocessor
        sync_frames = model_dict['syncformer_preprocess_tensor'](frames_sync_tensor).unsqueeze(0).to(device)
        
        model_dict_with_device = {**model_dict, 'device': device}
        sync_feat = encode_video_with_sync(sync_frames, AttributeDict(model_dict_with_device))

        prompts = [negative_prompt, prompt]
        text_feat_res, _ = encode_text_feat(prompts, AttributeDict(model_dict_with_device))
        text_feat, uncond_text_feat = text_feat_res[1:], text_feat_res[:1]
        if cfg.model_config.model_kwargs.text_length < text_feat.shape[1]:
            text_feat, uncond_text_feat = text_feat[:, :cfg.model_config.model_kwargs.text_length], uncond_text_feat[:, :cfg.model_config.model_kwargs.text_length]
        
        logging.info(f"Generating audio ({audio_len_in_s:.2f}s)...")
        
        latents = denoise_process(
            AttributeDict({'siglip2_feat': siglip_feat, 'syncformer_feat': sync_feat}), 
            AttributeDict({'text_feat': text_feat, 'uncond_text_feat': uncond_text_feat}), 
            audio_len_in_s, 
            AttributeDict({'foley_model': model_dict['foley_model'], 'device': device}), 
            cfg, guidance_scale, steps
        )
        
        return io.NodeOutput({"samples": latents.cpu(), "audio_len_in_s": audio_len_in_s})


class DACHunyuanVAEDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DACHunyuanVAEDecode",
            display_name="Hunyuan-Foley VAE Decode",
            category="HunyuanVideo-Foley",
            inputs=[
                io.Latent.Input("samples"),
                io.Vae.Input("vae"),
                io.Boolean.Input("unload_models_after_use", default=False, optional=True)
            ],
            outputs=[
                io.Audio.Output()
            ]
        )

    @classmethod
    def execute(cls, samples, vae, unload_models_after_use=False) -> io.NodeOutput:
        latents = samples["samples"]
        audio_len_in_s = samples["audio_len_in_s"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
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

            if unload_models_after_use:
                unload_foley_models()

        return io.NodeOutput(audio_out)

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
