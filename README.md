# ComfyUI HunyuanVideo-Foley 🎵

Generate high-fidelity, synchronized foley audio for any video directly within ComfyUI, powered by Tencent's HunyuanVideo-Foley model.

This custom node set provides a modular and offline-capable workflow for AI sound effect generation.

![image](https://github.com/BobRandomNumber/ComfyUI-HunyuanVideo_Foley/blob/main/example_workflows/Hunyuan-Foley.png)

---

## ✨ Features

-   **High-Fidelity Audio:** Generates 48kHz stereo audio using the advanced DAC VAE.
-   **Video-to-Audio Synchronization:** Leverages the Synchformer model to ensure audio events are timed with visual actions.
-   **Text-Guided Control:** Use text prompts, powered by the CLAP model, to creatively direct the type of sound you want to generate.
-   **Modular:** The workflow is broken into logical `Loader`, `Sampler`, and `VAE Decode` nodes, mirroring the standard Stable Diffusion workflow.
-   **VRAM Management:** Caches models in VRAM for fast, repeated generations. Includes an optional "Low VRAM" mode to unload models after use, ideal for memory-constrained systems.
-   **Offline Capable:** No automatic model downloads. Once you've downloaded the models, the node works entirely offline.

## ⚙️ Installation

### Method 1: Using ComfyUI Manager (Recommended)

1.  Open ComfyUI Manager.
2.  Click on `Install Custom Nodes`.
3.  Search for `ComfyUI-HunyuanVideo_Foley` and click `Install`.
4.  Restart ComfyUI.
5.  Follow the **Download Models** instructions below.

### Method 2: Manual Installation

1.  Navigate to your ComfyUI `custom_nodes` directory.
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  Clone this repository:
    ```bash
    git clone https://github.com/BobRandomNumber/ComfyUI-HunyuanVideo_Foley.git
    ```
3.  Install the required dependencies:
    ```bash
    cd ComfyUI-HunyuanVideo-Foley/
    pip install -r requirements.txt
    ```
4.  Restart ComfyUI.

---

## 📥 Download Models (Crucial Step)

This node requires you to download the model files manually and organize them in a specific folder structure. This ensures the node works offline and gives you full control.

1.  Navigate to `ComfyUI/models/`.
2.  Create a new folder named `hunyuan_foley`.

4.  **Download the following and place them inside your `hunyuan_foley` folder**

    *   **Hunyuan-Foley Base Models** from [Tencent/HunyuanVideo-Foley on Hugging Face](https://huggingface.co/tencent/HunyuanVideo-Foley/tree/main):
        *   `hunyuanvideo_foley.pth`
        *   `synchformer_state_dict.pth`
        *   `vae_128d_48k.pth`

    *   **SigLIP Vision Model** from [google/siglip2-base-patch16-512 on Hugging Face](https://huggingface.co/google/siglip2-base-patch16-512/tree/main):
        *   Create a new folder named `siglip2`.
        *   Download `model.safetensors`, `config.json` and `preprocessor_config.json` place them inside the `siglip2` folder.

    *   **CLAP Text Model** from [laion/larger_clap_general on Hugging Face](https://huggingface.co/laion/larger_clap_general/tree/main):
        *   Create a new folder named `clap`.
        *   Download `pytorch_model.safetensors`, `config.json`, `merges.txt` and `vocab.json` and place them inside the `clap` folder.

**Your final folder structure should look exactly like this:**

```
ComfyUI/
└── models/
    └── hunyuan_foley/        <-- You will see this folder selected in the Loader node 
        ├── hunyuanvideo_foley.pth
        ├── synchformer_state_dict.pth
        ├── vae_128d_48k.pth
        │
        ├── siglip2/          <-- Subfolder for SigLIP2
        │   ├── model.safetensors
        │   ├── config.json
        │   └── preprocessor_config.json
        │
        └── clap/             <-- Subfolder for CLAP
            ├── pytorch_model.safetensors
            ├── config.json
            ├── merges.txt
            └── vocab.json
```

## 🚀 Usage & Nodes

The workflow is designed to be modular and familiar to ComfyUI users.

### 1. `Hunyuan-Foley model loader`
This node loads the main diffusion model and all necessary conditioning models (SigLIP2, CLAP, Synchformer) into VRAM. These models are cached for fast subsequent generations.
-   **Inputs:**
    -   `model_path_name`: The model folder you created.
-   **Outputs:**
    -   `FOLEY_MODEL`: The loaded models, ready to be passed to the sampler.

### 2. `Hunyuan-Foley VAE loader`
This node loads the specialized DAC audio VAE used for decoding the final sound. Keeping it separate saves VRAM during the sampling process.
-   **Inputs:**
    -   `vae_name`: A dropdown to select the `vae_128d_48k.pth` file. It will search your `hunyuan_foley` model folder.
-   **Outputs:**
    -   `VAE`: The loaded DAC VAE model.

### 3. `Hunyuan-Foley Sampler`
This is the core node where the audio generation happens. It takes the video and text prompt and generates a latent representation of the audio.
-   **Inputs:**
    -   `foley_model`: The model from the `Hunyuan-Foley model loader`.
    -   `video_path`: A string path to your video file.
    -   `prompt`: Your text description of the desired sound.
    -   `guidance_scale`, `steps`, `seed`: Standard diffusion sampling parameters.
-   **Outputs:**
    -   `LATENT`: A latent tensor representing the generated audio. This is passed to the VAE Decode node.

### 4. `Hunyuan-Foley VAE Decode`
This node takes the latent tensor from the sampler and converts it into a final audio waveform. It also contains the VRAM management toggle.
-   **Inputs:**
    -   `samples`: The `LATENT` output from the `Hunyuan-Foley Sampler`.
    -   `vae`: The `VAE` output from the `Hunyuan-Foley VAE loader`.
    -   `unload_models_after_use` (Boolean Toggle):
        -   **`False` (Default):** Keeps the main models in VRAM for fast subsequent generations.
        -   **`True` (Low VRAM Mode):** Frees VRAM by moving the main models to system RAM after generation is complete. The next generation will be slower as it requires a full reload.
-   **Outputs:**
    -   `AUDIO`: The final audio waveform, which can be connected to `Save Audio`, `Preview Audio`, or a `Video Combine` node.

### Performance & Workflow Tips
-   **VRAM Requirement:** For the best performance (keeping models cached), a GPU with approximately **10-12GB of VRAM** is recommended.
-   **Initial Load:** The first time you run a workflow, the `Hunyuan-Foley model loader` will take a moment to load all models from disk into VRAM. Subsequent runs in the same session will be faster as long as models are not unloaded.
-   **Low VRAM Mode:** If you are running low on VRAM or only need to generate a single audio track, set the `unload_models_after_use` toggle on the `Hunyuan-Foley VAE Decode` node to `True`. This will significantly reduce the idle VRAM footprint after the workflow completes.

## 🙏 Acknowledgements

-   **Tencent Hunyuan:** For creating and open-sourcing the original [HunyuanVideo-Foley](https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley) model.
-   **Google Research:** For the SigLIP model.
-   **LAION:** For the CLAP model.
-   **Descript:** For the [descript-audio-codec](https://github.com/descriptinc/descript-audio-codec) (DAC VAE).

-   **v-iashin:** For the [Synchformer](https://github.com/v-iashin/Synchformer) model.


