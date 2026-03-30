# LoRA Adapters

Place your LoRA adapters here. Two formats are supported:

- **PEFT directory**: a folder containing `adapter_model.safetensors` + `adapter_config.json`
- **ComfyUI single file**: a `.safetensors` file with alpha baked in (no config needed)

Point the server to this directory:

```bash
./build/ace-server --models ./models --loras ./loras
```
