import argparse
from pathlib import Path
import os
import re

import numpy as np
import torch


def _parse_str_field(run_log_text: str, key: str) -> str | None:
    # Examples:
    # "prompt_0": "A photo of an object"
    # "save_lora_dir": "/root/.../lora",
    m = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]*)"', run_log_text)
    if not m:
        return None
    return m.group(1)


def _parse_bool_field(run_log_text: str, key: str) -> bool | None:
    m = re.search(rf'"{re.escape(key)}"\s*:\s*(true|false)', run_log_text)
    if not m:
        return None
    return m.group(1) == "true"


def _parse_num_field(run_log_text: str, key: str) -> float | int | None:
    # Handles ints/floats in JSON-ish log.
    m = re.search(rf'"{re.escape(key)}"\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', run_log_text)
    if not m:
        return None
    s = m.group(1)
    if "." in s or "e" in s.lower():
        return float(s)
    return int(s)


def _parse_alpha_list(run_log_text: str) -> list[float]:
    # Prefer rescheduled alpha_list
    m = re.search(r'Rescheduled alpha_list=\[\s*([0-9eE\.\,\s\-\+]+?)\s*\]', run_log_text)
    if m:
        items = [x for x in re.split(r'[\s,]+', m.group(1).strip()) if x]
        return [float(x) for x in items]

    m = re.search(r'Alpha_list=\[\s*([0-9eE\.\,\s\-\+]+?)\s*\]', run_log_text)
    if not m:
        raise RuntimeError("无法从 run.log 解析 alpha_list")
    items = [x for x in re.split(r'[\s,]+', m.group(1).strip()) if x]
    return [float(x) for x in items]


def _load_run_params(run_dir: Path) -> dict:
    log_path = run_dir / "diffmorpher_run.log"
    if not log_path.exists():
        raise FileNotFoundError(f"找不到: {log_path}")
    txt = log_path.read_text(encoding="utf-8", errors="ignore")

    img0 = _parse_str_field(txt, "img_path_0")
    img1 = _parse_str_field(txt, "img_path_1")
    prompt0 = _parse_str_field(txt, "prompt_0")
    prompt1 = _parse_str_field(txt, "prompt_1")
    guidance_scale = _parse_num_field(txt, "guidance_scale")
    num_inference_steps = _parse_num_field(txt, "num_inference_steps")
    use_lora = _parse_bool_field(txt, "use_lora")
    use_adain = _parse_bool_field(txt, "use_adain")
    save_lora_dir = _parse_str_field(txt, "save_lora_dir")

    alpha_list = _parse_alpha_list(txt)

    missing = []
    for k, v in [
        ("img_path_0", img0),
        ("img_path_1", img1),
        ("prompt_0", prompt0),
        ("prompt_1", prompt1),
        ("guidance_scale", guidance_scale),
        ("num_inference_steps", num_inference_steps),
        ("use_lora", use_lora),
        ("use_adain", use_adain),
        ("save_lora_dir", save_lora_dir),
    ]:
        if v is None:
            missing.append(k)

    if missing:
        raise RuntimeError(f"run.log 缺少字段或解析失败: {missing}")

    return {
        "img_path_0": Path(img0),
        "img_path_1": Path(img1),
        "prompt_0": prompt0,
        "prompt_1": prompt1,
        "guidance_scale": float(guidance_scale),
        "num_inference_steps": int(num_inference_steps),
        "use_lora": bool(use_lora),
        "use_adain": bool(use_adain),
        "save_lora_dir": Path(save_lora_dir),
        "alpha_list": alpha_list,
    }


def _norm_flat(x: torch.Tensor) -> float:
    return float(torch.linalg.norm(x.reshape(-1)).detach().cpu().item())


def _cosine_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.reshape(-1)
    bb = b.reshape(-1)
    cos = (aa @ bb) / (torch.linalg.norm(aa) * torch.linalg.norm(bb) + 1e-12)
    return float(cos.detach().cpu().item())


def main():
    parser = argparse.ArgumentParser(description="检查 DiffMorpher 的 img_noise 是否近似同一球面。")
    parser.add_argument("--run_dir", required=True, type=str, help="如 results/2026-03-19_03")
    parser.add_argument("--diffmorpher_dir", default=str(Path(__file__).resolve().parents[1] / "diffmorpher"), type=str)
    parser.add_argument("--ckpt_name", default="v2-1_512-ema-pruned.ckpt")
    parser.add_argument("--cfg_name", default="v2-inference.yaml")
    parser.add_argument("--device", default="auto", type=str, help="auto/cpu/cuda/cuda:N")
    parser.add_argument("--num_inversion_steps_override", default=0, type=int,
                        help="0 表示使用 run.log 里的 num_inference_steps")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    params = _load_run_params(run_dir)

    # resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device.startswith("cuda") and torch.cuda.is_available():
        if device == "cuda":
            dev = torch.device("cuda:0")
        else:
            idx = int(device.split(":", 1)[1])
            dev = torch.device(f"cuda:{idx}")
        torch.cuda.set_device(dev.index if hasattr(dev, "index") and dev.index is not None else 0)
    else:
        dev = torch.device(device)

    diffmorpher_dir = Path(args.diffmorpher_dir)
    ckpt_path = diffmorpher_dir / args.ckpt_name
    cfg_path = diffmorpher_dir / args.cfg_name

    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到 ckpt: {ckpt_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"未找到 cfg: {cfg_path}")

    # import local DiffMorpherPipeline + utilities
    import sys

    if str(diffmorpher_dir) not in sys.path:
        sys.path.insert(0, str(diffmorpher_dir))
    if str(diffmorpher_dir.parent) not in sys.path:
        sys.path.insert(0, str(diffmorpher_dir.parent))

    # Build pipeline (same components as pipeline.py)
    from diffusers import StableDiffusionPipeline
    from model import DiffMorpherPipeline  # type: ignore
    from utils.model_utils import get_img, slerp  # type: ignore
    from utils.lora_utils import load_lora  # type: ignore

    # Keep local relative path behavior consistent
    old_cwd = Path.cwd()
    try:
        os.chdir(str(diffmorpher_dir))
        sd_pipe = StableDiffusionPipeline.from_single_file(
            str(ckpt_path),
            original_config_file=str(cfg_path),
            torch_dtype=torch.float32,
            local_files_only=True,
        )
    finally:
        os.chdir(str(old_cwd))

    pipe = DiffMorpherPipeline(
        vae=sd_pipe.vae,
        text_encoder=sd_pipe.text_encoder,
        tokenizer=sd_pipe.tokenizer,
        unet=sd_pipe.unet,
        scheduler=sd_pipe.scheduler,
        safety_checker=getattr(sd_pipe, "safety_checker", None),
        feature_extractor=getattr(sd_pipe, "feature_extractor", None),
    )
    pipe.to(dev)
    pipe._run_device = dev

    # use same inference steps as run (or override)
    num_inference_steps = params["num_inference_steps"]
    if args.num_inversion_steps_override and args.num_inversion_steps_override > 0:
        num_inference_steps = args.num_inversion_steps_override

    pipe.scheduler.set_timesteps(num_inference_steps)
    pipe.use_lora = params["use_lora"]
    pipe.use_adain = params["use_adain"]

    img0_path = params["img_path_0"]
    img1_path = params["img_path_1"]
    prompt0 = params["prompt_0"]
    prompt1 = params["prompt_1"]
    guidance_scale = params["guidance_scale"]

    # guidance_scale<=1 时 cond 里不会拼 unconditional embedding
    batch_size = 1
    text_embeddings_0 = pipe.get_text_embeddings(prompt0, guidance_scale, None, batch_size)
    text_embeddings_1 = pipe.get_text_embeddings(prompt1, guidance_scale, None, batch_size)

    # Note: get_img in utils/model_utils.py takes PIL/nparray and normalizes.
    # We want to align with pipeline/model.py: image2latent is called with img_noise inputs that are already get_img(image) tensors.
    from PIL import Image

    img0_pil = Image.open(img0_path).convert("RGB")
    img1_pil = Image.open(img1_path).convert("RGB")
    img0_t = get_img(img0_pil)  # [1,3,512,512]
    img1_t = get_img(img1_pil)

    # load lora (if enabled) from run_dir/lora/morph_lora_{0,1}.ckpt
    lora_0 = None
    lora_1 = None
    morph_name = (run_dir / "morph").name  # likely "morph"
    lora_0_path = params["save_lora_dir"] / f"{morph_name}_lora_0.ckpt"
    lora_1_path = params["save_lora_dir"] / f"{morph_name}_lora_1.ckpt"

    if pipe.use_lora:
        if not lora_0_path.exists():
            raise FileNotFoundError(f"找不到 LoRA0: {lora_0_path}")
        if not lora_1_path.exists():
            raise FileNotFoundError(f"找不到 LoRA1: {lora_1_path}")
        lora_0 = torch.load(str(lora_0_path), map_location="cpu")
        lora_1 = torch.load(str(lora_1_path), map_location="cpu")

    # compute img_noise_0
    if pipe.use_lora:
        pipe.unet = load_lora(pipe.unet, lora_0, lora_1, 0.0)
    img_noise_0 = pipe.ddim_inversion(pipe.image2latent(img0_t), text_embeddings_0)

    # compute img_noise_1
    if pipe.use_lora:
        pipe.unet = load_lora(pipe.unet, lora_0, lora_1, 1.0)
    img_noise_1 = pipe.ddim_inversion(pipe.image2latent(img1_t), text_embeddings_1)

    R0 = _norm_flat(img_noise_0)
    R1 = _norm_flat(img_noise_1)
    end_delta = abs(R0 - R1) / (0.5 * (R0 + R1) + 1e-12)
    cos01 = _cosine_flat(img_noise_0, img_noise_1)

    print("=== Endpoint check (img_noise_0/img_noise_1) ===")
    print(f"R0=||img_noise_0||={R0:.6f}")
    print(f"R1=||img_noise_1||={R1:.6f}")
    print(f"relative_radius_diff(delta)={end_delta*100:.4f}%")
    print(f"cosine(img_noise_0,img_noise_1)={cos01:.6f}")
    print()

    alpha_list = params["alpha_list"]

    def check_for_adain(adain_flag: bool) -> dict:
        radii = []
        for a in alpha_list:
            # slerp expects torch tensors with correct shape, and a scalar alpha
            lat = slerp(img_noise_0, img_noise_1, float(a), adain=adain_flag)
            radii.append(_norm_flat(lat))
        radii = np.array(radii, dtype=np.float64)
        return {
            "adain": adain_flag,
            "radius_mean": float(radii.mean()),
            "radius_std": float(radii.std(ddof=0)),
            "radius_rel_std": float(radii.std(ddof=0) / (radii.mean() + 1e-12)),
            "radius_min": float(radii.min()),
            "radius_max": float(radii.max()),
        }

    print("=== Trajectory check (slerp along alpha_list) ===")
    stats_no_adain = check_for_adain(adain_flag=False)
    stats_with_adain = check_for_adain(adain_flag=True)

    for st in [stats_no_adain, stats_with_adain]:
        print(f"adain={st['adain']}: mean={st['radius_mean']:.6f} "
              f"std={st['radius_std']:.6f} rel_std={st['radius_rel_std']*100:.4f}% "
              f"min={st['radius_min']:.6f} max={st['radius_max']:.6f}")

    # Heuristic verdict
    # With adain=False we expect near-constant radius if endpoints are on same sphere.
    rel = stats_no_adain["radius_rel_std"]
    if end_delta < 0.05 and rel < 0.05:
        verdict = "更接近同一球面（建议以 adain=False 的结果为准）"
    else:
        verdict = "不太像同一球面（或者 adain 会破坏球面约束）"

    print()
    print("Verdict:", verdict)


if __name__ == "__main__":
    main()

