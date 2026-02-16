from comfy_api.latest import ComfyExtension, io
from .hunyuan_foley import HunyuanFoleyModelLoader, LoadDACHunyuanVAE, HunyuanFoleySampler, DACHunyuanVAEDecode

class HunyuanFoleyExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            HunyuanFoleyModelLoader,
            LoadDACHunyuanVAE,
            HunyuanFoleySampler,
            DACHunyuanVAEDecode,
        ]

async def comfy_entrypoint() -> HunyuanFoleyExtension:
    return HunyuanFoleyExtension()
