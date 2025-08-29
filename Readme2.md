That error means you tried to move a meta tensor (a placeholder with no storage) out of the meta device. cpu_offload() (and any real .to("cuda")) needs a real tensor (on CPU) to copy from — meta tensors have no underlying data, so the copy fails: “Cannot copy out of meta tensor — No data.”

Why that happened in your run
	•	You used init_empty_weights() (or the model was left in a not-yet-materialized state), so many parameters are still on meta.
	•	You attached cpu_offload() before actually materializing/loading the weights onto CPU, so the hook tried to move a meta tensor to GPU and crashed.

How to fix it (short answer)
	•	Materialize weights on CPU first, then attach cpu_offload() hooks to expert modules. In practice: load the checkpoint with device_map="cpu" (or load_checkpoint_and_dispatch(..., device_map="cpu")) so parameters become real CPU tensors, verify there are no meta tensors, and then add the cpu_offload hooks.

Important note: materializing all weights on CPU requires enough host RAM (and/or offload storage). If your host RAM is limited, use the Accelerate offload_folder option so the state dict pages to disk while you keep the live tensors on CPU.

Working code snippet (replace your current flow with this)

# Assumes: transformers, accelerate, torch installed
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate.hooks import cpu_offload
import torch, re, os, contextlib

MODEL_ID = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
DEVICE = "cuda"
offload_folder = "./offload_qwen_moe"
os.makedirs(offload_folder, exist_ok=True)
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# 1) Build empty model graph (optional, but useful to avoid huge CPU spike)
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config, torch_dtype=DTYPE, trust_remote_code=True)

# 2) Materialize weights onto CPU (important: not meta!)
#    Use device_map="cpu" so everything gets loaded to real CPU tensors.
model = load_checkpoint_and_dispatch(
    model,
    MODEL_ID,
    device_map="cpu",            # force parameters to CPU (real tensors)
    offload_folder=offload_folder,
    offload_state_dict=False,    # keep tensors materialized in model (not only state dict)
    dtype=DTYPE
)

# 3) Verify no meta tensors remain
meta_found = any(p.device.type == "meta" for p in model.parameters())
assert not meta_found, "Some parameters are still meta. Ensure load_checkpoint_and_dispatch completed."

# 4) Find expert submodules (heuristic) and attach cpu_offload hooks
expert_name_re = re.compile(r"\.experts(\.|$)|\bmoe\b|\bmixeral\b", re.IGNORECASE)
expert_modules = []
for name, module in model.named_modules():
    # heuristic: names containing 'experts' or 'moe'
    if expert_name_re.search(name):
        expert_modules.append((name, module))

print("Experts found:", len(expert_modules))

# Attach offload hook AFTER weights are real (CPU)
for name, module in expert_modules:
    # Move each expert to CPU (should already be CPU) then attach hook to move to GPU during forward
    module.to("cpu")
    cpu_offload(module, execution_device=DEVICE, offload_buffers=True)

# 5) Keep small shared/router parts on GPU (optional)
with contextlib.suppress(Exception):
    # push embeddings, final layernorm/lm_head etc to GPU
    for n, m in model.named_modules():
        if "experts" not in n and ("embed" in n or "lm_head" in n or "norm" in n or "router" in n):
            try:
                m.to(DEVICE)
            except Exception:
                pass

torch.cuda.empty_cache()

Why this order matters
	•	init_empty_weights() creates parameter placeholders (meta).
	•	load_checkpoint_and_dispatch(..., device_map="cpu", offload_state_dict=False) replaces meta tensors with real CPU tensors.
	•	Only after parameters exist on CPU can cpu_offload() reliably move them to GPU during forward and back off after.

If you used load_checkpoint_and_dispatch(..., device_map="auto") previously, Accelerate may have already put many experts on CUDA, or left others as meta — that mixed state can be confusing. For strict “keep experts on CPU until needed” behavior, force device_map="cpu" when materializing.

Extra debugging tips
	•	Check for meta tensors:

meta_params = [(n,p.device) for n,p in model.named_parameters() if p.device.type == "meta"]
print("Meta params:", len(meta_params))

	•	Check how much GPU memory is used after attaching hooks but before generation to ensure only router/shared layers are on GPU.
	•	Temporarily set use_cache=False in generate() to reduce KV cache noise when measuring model-weight VRAM.

If you want, I can:
	•	Give a version that uses disk-backed offload (if RAM shortage is a problem).
	•	Add an LRU cache wrapper for experts (keep most recently used experts on GPU) to reduce per-token transfer overhead.
	•	Tailor the expert-name regex to Qwen3’s exact module names (I can craft it if you paste list(model.named_modules())[:200] output or a few example names).
