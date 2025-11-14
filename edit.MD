Awesome — you answered the important constraints, so I built a concrete end-to-end pipeline that uses only Qwen-Image-Edit (no extra ML models) to produce a clean, per-category binary mask (black background, white object) from an (image + simple text prompt). The pipeline relies on prompt engineering, iterative edit passes, multi-scale tiling for tiny objects, and classic image-processing postprocessing (thresholding, morphological ops, CRF-style refinement) to drive the diffusion editor toward a discrete mask output and then extract it. Because the model is not designed to output masks, we treat it as a painter that we coerce into painting a pure mask through instructions + controlled editing passes.

Below you’ll find:
	1.	Overview (short)
	2.	Full pipeline (step-by-step)
	3.	Prompt templates and recommended parameters
	4.	Multi-scale & tiling method for tiny objects
	5.	Postprocessing recipe (to get crisp binary mask)
	6.	Iterative refinement strategy (to reach near-perfect segmentation)
	7.	Pseudocode / implementation outline (Python)
	8.	Limitations and practical tips

⸻

1) Overview (one line)

Use Qwen-Image-Edit as a controllable painter: 1) force it to paint the object pure white and everything else pure black through one or more targeted edit passes, 2) threshold/clean the painted image to a strict binary mask, 3) refine with morphological and pixel-edge guided filtering, using multi-scale crops to capture tiny parts.

⸻

2) Full pipeline (step-by-step)

Input: RGB image I, user prompt P (simple, e.g. “car”, “person”, “green pot”)
Output: Binary mask M (same resolution as I; object = 255 white, background = 0 black)

A. Preprocessing
	1.	Resize image to model-preferred resolution (preserve aspect ratio) — keep original for final mapping. If model accepts multiple sizes, choose the max resolution it supports; otherwise use a square (e.g., 1024) with padding.
	2.	Normalize / convert color if required by the inference call.

B. Primary mask generation (single-pass attempt)
3. Craft a highly prescriptive prompt instructing Qwen-Image-Edit to paint a pure binary mask: see prompts section.
4. Run Qwen-Image-Edit once with strong guidance (high CFG) and deterministic seed if possible. Request output image. This produces O1 (expected black/white-like image, possibly antialiased/grayscale).

C. Postprocess O1 to binary mask
5. Convert O1 to grayscale, compute threshold (Otsu or fixed like 128) → binary B1.
6. Morphological clean: remove tiny islands, fill holes, use connected components to keep components consistent with expected size/focus.
7. If B1 is acceptable (quality rules — e.g., IoU heuristics vs object area from prompt heuristics), return as final M. If not, continue.

D. Iterative refinement with targeted editing (if single pass insufficient)
8. Feed B1 back into the model as an input mask for a refinement edit. Use a new prompt: “Refine the white shape to exactly match the [object] silhouette; keep background pure black; sharpen edges; no textures.” In edit mode give B1 as the region to preserve/paint. That forces the model to repaint inside/outside region. This produces O2.
9. Threshold + morphological clean → B2. If good → M.

E. Multi-scale tiling for small objects / very fine detail
10. If object contains tiny components or small details (green pot, tiny handles), run a tiling pass: split the image into overlapping crops (e.g., 512 px crops with 25–33% overlap). For each crop containing the object area (or run over full grid if uncertain), repeat steps B–D to create crop masks, then merge via weighted overlap (use max or union). Upsample and stitch with blending and final threshold.

F. Final edge refinement
11. One last pass of morphological smoothing + guided edge preservation (use original RGB as guide) — e.g., bilateral filter on mask edge and rethreshold or use DenseCRF-style unary from mask + pairwise from RGB to refine border alignment. Output final mask M.

⸻

3) Prompt templates & parameters

These are the most important levers: extremely explicit, deterministic wording plus negatives. Use the exact style.

Primary mask prompt (single pass):

“Create a binary segmentation mask image of the [OBJECT] in this photo. The output image must be only two flat colors: black background (RGB 0,0,0) and the object filled white (RGB 255,255,255). No shading, no gray, no overlay, no transparency, no textures, no outlines — only sharp, perfectly filled white silhouette of the [OBJECT]. Preserve exact object boundary; do not add decorations or cut parts. Return a pure black-and-white mask with crisp edges.”

Refinement prompt (using generated mask as guidance):

“Refine this white silhouette so its boundaries exactly match the [OBJECT] in the original image. Keep the background fully black and the object fully white. Remove stray white pixels and fill holes in the object. Make contours precise — no smoothing blur, no color leakage.”

Negative prompt (if API supports):

“No color, no shading, no overlay, no texture, no shadows, no partial transparency, no outlines, no additional objects, no text.”

Parameters (recommended):
	•	seed: fixed when you need deterministic outputs (set seed across runs)
	•	cfg_scale / true_cfg_scale: high-ish, e.g., 7–12 (forces adherence to prompt)
	•	steps: moderate (40–100) — higher helps crispness for diffusion
	•	sampler: FlowMatch Euler (or whatever denoiser the model docs recommend)
	•	n: 1 output per attempt
	•	negative_prompt: use above if supported

⸻

4) Multi-scale & tiling for tiny/finely detailed objects

Tiny details are the hardest part. Without adding an external detector, use a multi-scale tile strategy:
	1.	Run single-pass on full image → get B_full.
	2.	If B_full misses tiny components, identify candidate regions to crop:
	•	Option A: Slide window dense grid with high overlap across the whole image.
	•	Option B: Use connected components of an initial low-thresholded result to find candidate centers to zoom.
	3.	For each crop at a higher native model resolution, run the primary prompt again but targeted (prompt includes crop context “object in this crop: [object]”).
	4.	Merge masks back to original coordinates using union. Use feathering near crop boundaries and final threshold.

This recovers thin stems, small pots, shoe laces, etc.

⸻

5) Postprocessing recipe (to get crisp binary mask)

After you get the painted mask image O from the model:
	1.	Convert to grayscale: G = rgb2gray(O).
	2.	Thresholding:
	•	Use Otsu if model painting varies; else use fixed 128 threshold: B = G > 128.
	•	To enforce strict black/white, set values to 0 or 255.
	3.	Morphological operations:
	•	B = remove_small_objects(B, min_size=MIN_PIXELS) (drop specks).
	•	B = binary_closing(B, selem=disk(3)) to fill holes.
	•	B = binary_opening(B, selem=disk(1)) to remove small spurs.
	4.	Connected component filtering:
	•	If user asks for a single instance: choose the component that best matches expected size or location (center-most or largest).
	5.	Edge refinement (optional but recommended):
	•	Use DenseCRF (pydensecrf) with unary from mask (high confidence for white vs black) and pairwise terms from original RGB to snap edges to color boundaries. If you prefer not to install DenseCRF, an edge-aware bilateral filter + re-threshold works too.
	6.	Upsample/Downsample: If you used padded/resized image for model input, remap mask to original full resolution using nearest-neighbor sampling to preserve crisp edges.

⸻

6) Iterative refinement strategy (converging to perfect)

Because Qwen-Image-Edit is generative, we rely on multiple passes to converge:
	1.	Pass 1 (coarse): Full image prompt → produce O1 → B1.
	2.	Pass 2 (refine): Feed B1 back as mask input region and use refinement prompt → O2 → B2.
	3.	Pass 3 (zoom tiny areas): Run tiled zoom passes where B2 shows missing small pieces; merge results.
	4.	Pass 4 (final polish): Use final prompt “Make mask binary only, sharpen edges” with B_merged as initial mask for last repaint. Then final threshold.

This loop should be automated with a stopping criterion: e.g., pixel difference between B_k and B_{k-1} < small epsilon or fixed number of iterations (3–4).

⸻

7) Implementation outline (Python pseudocode)

Below is an executable-style pseudocode. Integrate with your HF / local Qwen-Image-Edit inference call where qwen_edit(...) is the wrapper that calls the model and returns PIL image(s). Replace with your actual API call.

import numpy as np
from PIL import Image
import cv2
from skimage.morphology import remove_small_objects, binary_closing, binary_opening, disk
# Optional: import pydensecrf

# --- helper functions ---
def to_grayscale(img_pil):
    arr = np.array(img_pil.convert("L"))
    return arr

def threshold_mask(gray, method="fixed", fixed=128):
    if method == "otsu":
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th
    else:
        _, th = cv2.threshold(gray, fixed, 255, cv2.THRESH_BINARY)
        return th

def clean_mask(binary_uint8, min_size=500, closing_disk=3):
    bin_bool = binary_uint8.astype(bool)
    bin_bool = remove_small_objects(bin_bool, min_size=min_size)
    bin_bool = binary_closing(bin_bool, selem=disk(closing_disk))
    bin_bool = binary_opening(bin_bool, selem=disk(1))
    return (bin_bool.astype(np.uint8) * 255)

def qwen_edit(image_pil, prompt, negative_prompt=None, seed=None, cfg_scale=10, steps=60, mask_in=None):
    # Placeholder: call your Qwen-Image-Edit inference here
    # Return PIL.Image
    raise NotImplementedError("Hook Qwen-Image-Edit inference here.")

# --- pipeline ---
def generate_mask_with_qwen(image_pil, object_name, config):
    # 1) primary pass
    primary_prompt = f"Create a binary segmentation mask image of the {object_name}. "\
                     "Output must be two flat colors: black background (0,0,0) and white object (255,255,255). "\
                     "No shading, no textures, no outlines. Exactly filled white silhouette."
    output1 = qwen_edit(image_pil, prompt=primary_prompt,
                        negative_prompt=config.get("neg_prompt"),
                        seed=config.get("seed"), cfg_scale=config.get("cfg"), steps=config.get("steps"))
    gray1 = to_grayscale(output1)
    bin1 = threshold_mask(gray1, method="fixed", fixed=config.get("thresh",128))
    clean1 = clean_mask(bin1, min_size=config.get("min_size",200))
    # 2) refinement pass using mask input
    refine_prompt = f"Refine this white silhouette to exactly match the {object_name}. Keep background black, the object white, sharpen edges."
    mask_in_pil = Image.fromarray(clean1).convert("RGB")
    output2 = qwen_edit(image_pil, prompt=refine_prompt,
                        negative_prompt=config.get("neg_prompt"),
                        seed=config.get("seed"), cfg_scale=config.get("cfg"),
                        steps=config.get("steps"), mask_in=mask_in_pil)
    gray2 = to_grayscale(output2)
    bin2 = threshold_mask(gray2, method="fixed", fixed=config.get("thresh",128))
    clean2 = clean_mask(bin2, min_size=config.get("min_size",200))
    # 3) optional: multi-scale tiling (if small parts missing) -> produce tile masks and union
    # 4) final: CRF / edge refine (optional)
    final_mask = clean2
    return final_mask

Notes about hooking qwen_edit(...):
	•	If you call Qwen locally (diffusers integration), pass image and prompt to the image-edit pipeline and set num_inference_steps, guidance_scale.
	•	If using Hugging Face Inference API, construct payload accordingly and get back the edited image as base64.
	•	Provide mask_in (the current binary) for refinement edits (Qwen-Image-Edit supports image+mask editing modes).

⸻

8) Limitations, practical realities & tips
	1.	Model not designed for masks: Qwen-Image-Edit was built to produce edited images, not binary masks. We rely on its ability to follow instructions to paint a mask. This will work well many times, but is not as robust as a dedicated segmentation network in worst-case scenarios (thin hair, transparent/occluded objects). The iterative + tiling recipe mitigates much of this.
	2.	Determinism & seeds: Diffusion is stochastic; fix seed for repeatability. Different seeds sometimes produce slightly different boundaries; you can ensemble outputs (union/intersection) to stabilize.
	3.	CFG tuning: If model ignores prompt, raise cfg_scale. If model produces unnatural mask artifacts, lower it. Typical range: 7–12.
	4.	Tiny objects: Use high-resolution crops (tile the image and run the same process). This is the single most important step to capture tiny details like a “green pot”.
	5.	Runtime cost: Multiple passes + tiles = heavier compute. Optimize by first testing full-image single-pass and only tile where needed.
	6.	Edge snapping: CRF or guided edge refinement with original RGB improves final border alignment significantly. This is classic postprocessing, not an external neural model.
	7.	Quality monitoring: Use heuristics such as mask coverage ratio (object area / image area) or run a small human-in-the-loop step to verify when quality must be perfect.
	8.	Color names / small color-based items: For color-specific objects like “green pot”, include color in the prompt (“mask the green pot, not other pots”) — Qwen’s VLM will use color cues.

⸻

Quick troubleshooting recipes
	•	Mask is gray/antialiased: use fixed threshold 128 and morphological closing; if still fuzzy, re-run refinement pass and set cfg_scale higher.
	•	Parts missing (thin stems): run tiles at 2× resolution centered on missing area; union the results.
	•	Extra white islands: remove small components by min_size or keep only the N largest components if you expect one object.
	•	Edge overshoot into background: apply erosion by 1–2 px then CRF refinement with RGB.

⸻

Example parameter recommendations (start here)
	•	steps: 60
	•	cfg_scale: 9–11
	•	seed: fixed value for reproducibility
	•	threshold: 128
	•	min_size: set proportional to image size (e.g., 0.001 * image_area)
	•	tile size: 512 with 128 px overlap for high-res images

⸻

Final notes & next actions
	•	This pipeline deliberately uses only Qwen-Image-Edit for the semantic/visual reasoning and editing steps; all other steps are image-processing operations (thresholding, morphological ops, CRF).
	•	If you want, I can now:
	•	Produce a concrete ready-to-run Python script with placeholders for your Qwen inference call (Hugging Face local/diffusers or HF Inference API).
	•	Give tuned example prompts and parameter defaults for your model runtime.
	•	Provide an automated tiling/merge implementation and a CRF-based refinement snippet.

Tell me which of those you want next and I’ll generate the code (with the exact inference call style you use — local diffusers, HF inference API, or something else).