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
