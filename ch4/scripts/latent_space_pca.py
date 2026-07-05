import argparse
from pathlib import Path
import os
import re

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from PIL import Image
from diffusers import StableDiffusionPipeline
from torchvision import transforms


def parse_img_paths_from_runlog(run_dir: Path) -> tuple[Path | None, Path | None]:
    log_path = run_dir / "diffmorpher_run.log"
    if not log_path.exists():
        return None, None
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    # run.log stores something like:
    # "img_path_0": "/.../assets/banana1.png",
    m0 = re.search(r'"img_path_0"\s*:\s*"([^"]+)"', txt)
    m1 = re.search(r'"img_path_1"\s*:\s*"([^"]+)"', txt)
    img0 = Path(m0.group(1)) if m0 else None
    img1 = Path(m1.group(1)) if m1 else None
    return img0, img1


def load_vae(
    diffmorpher_dir: Path,
    ckpt_name: str,
    cfg_name: str,
    device: str,
) -> torch.nn.Module:
    ckpt_path = diffmorpher_dir / ckpt_name
    cfg_path = diffmorpher_dir / cfg_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到 ckpt: {ckpt_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"未找到 cfg: {cfg_path}")

    # 复用 pipeline.py 的加载逻辑：在 diffmorpher 目录下运行，命中本地相对路径。
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

    vae = sd_pipe.vae.to(device)
    vae.eval()
    return vae


def get_preprocess_512():
    # 对齐 diffmorpher/utils/model_utils.py:get_img
    return transforms.Compose(
        [
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


@torch.no_grad()
def encode_image_to_vae_latent_vector(
    vae: torch.nn.Module,
    img_path: Path,
    transform,
    device: str,
) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)  # [1, 3, 512, 512]

    lat = vae.encode(x)["latent_dist"].mean  # [1, 4, 64, 64]
    # 对齐 diffmorpher/model.py:image2latent
    lat = lat * 0.18215

    lat_np = lat[0].detach().cpu().numpy()
    return lat_np.reshape(-1)


def pca_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    简单 PCA（不依赖 sklearn）:
    X: [N, D]
    返回:
      coords: [N, 2]
      mean: [1, D]
      Vt2: [2, D]
    """
    X_mean = X.mean(axis=0, keepdims=True)
    Xc = X - X_mean
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ Vt[:2].T
    return coords, X_mean, Vt[:2]


def pca_3d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA 投影到前三个主成分：
    X: [N, D]
    返回 coords: [N, 3]
    """
    X_mean = X.mean(axis=0, keepdims=True)
    Xc = X - X_mean
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ Vt[:3].T
    return coords, X_mean, Vt[:3]


def try_make_plotly_3d(
    run_dir: Path,
    coords3: np.ndarray,
    morph_paths: list[Path],
    frame_alphas: np.ndarray,
) -> None:
    """
    导出可交互的 3D 图（HTML）。
    若 plotly 不存在则静默跳过。
    """
    try:
        import plotly.graph_objects as go
    except Exception:
        return

    coords_gen3 = coords3[2:]
    c_img03 = coords3[0]
    c_img13 = coords3[1]

    hover_text_gen = [p.stem for p in morph_paths]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=[float(c_img03[0])],
            y=[float(c_img03[1])],
            z=[float(c_img03[2])],
            mode="markers",
            marker=dict(size=6, color="black"),
            name="img0",
            hovertemplate="img0<br>PC1=%{x:.4f}<br>PC2=%{y:.4f}<br>PC3=%{z:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[float(c_img13[0])],
            y=[float(c_img13[1])],
            z=[float(c_img13[2])],
            mode="markers",
            marker=dict(size=6, color="black"),
            name="img1",
            hovertemplate="img1<br>PC1=%{x:.4f}<br>PC2=%{y:.4f}<br>PC3=%{z:.4f}<extra></extra>",
        )
    )

    # morph frames：按 alpha 映射到 Blues 颜色
    fig.add_trace(
        go.Scatter3d(
            x=coords_gen3[:, 0],
            y=coords_gen3[:, 1],
            z=coords_gen3[:, 2],
            mode="markers+lines",
            marker=dict(
                size=4,
                color=frame_alphas,
                colorscale="Blues",
                cmin=0.0,
                cmax=1.0,
                showscale=True,
                colorbar=dict(title="alpha"),
            ),
            line=dict(color="rgba(0,0,255,0.35)", width=2),
            name="morph frames",
            text=hover_text_gen,
            hovertemplate="frame %{text}<br>alpha=%{customdata:.3f}<br>PC1=%{x:.4f}<br>PC2=%{y:.4f}<br>PC3=%{z:.4f}<extra></extra>",
            customdata=frame_alphas,
        )
    )

    fig.update_layout(
        title=f"Latent-space trajectory (VAE latent PCA 3D)\\n{run_dir.name}",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(x=0.02, y=0.98),
    )

    out_html = run_dir / "latent_space_pca_3d_plotly.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print("saved:", out_html)


def main():
    parser = argparse.ArgumentParser(
        description="可视化 DiffMorpher 过程中 latent 空间轨迹（PCA 2D/3D）。"
    )
    parser.add_argument("--run_dir", required=True, type=str, help="包含 morph/ 和 diffmorpher_run.log 的目录")
    parser.add_argument("--img0", default="", type=str, help="原始图 0 路径（可留空，自动从 run.log 读取）")
    parser.add_argument("--img1", default="", type=str, help="原始图 1 路径（可留空，自动从 run.log 读取）")
    parser.add_argument("--diffmorpher_dir", default=str(Path(__file__).resolve().parents[1] / "diffmorpher"), type=str)
    parser.add_argument("--ckpt_name", default="v2-1_512-ema-pruned.ckpt")
    parser.add_argument("--cfg_name", default="v2-inference.yaml")
    parser.add_argument("--device", default="auto", type=str, help="auto/cpu/cuda/cuda:N")
    parser.add_argument("--elev", default=18.0, type=float, help="3D 图视角：仰角")
    parser.add_argument("--azim", default=-60.0, type=float, help="3D 图视角：方位角")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    morph_dir = run_dir / "morph"
    if not morph_dir.exists():
        raise FileNotFoundError(f"morph 目录不存在: {morph_dir}")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    img0_path = Path(args.img0) if args.img0.strip() else None
    img1_path = Path(args.img1) if args.img1.strip() else None
    if img0_path is None or img1_path is None:
        auto0, auto1 = parse_img_paths_from_runlog(run_dir)
        img0_path = auto0 if img0_path is None else img0_path
        img1_path = auto1 if img1_path is None else img1_path

    if img0_path is None or img1_path is None:
        raise RuntimeError("无法得到 img0/img1 路径。请在命令行显式传入 --img0/--img1，或确保 run.log 存在。")
    if not img0_path.exists():
        raise FileNotFoundError(f"img0 不存在: {img0_path}")
    if not img1_path.exists():
        raise FileNotFoundError(f"img1 不存在: {img1_path}")

    morph_paths = sorted(morph_dir.glob("*.png"), key=lambda p: int(p.stem))
    if not morph_paths:
        raise RuntimeError(f"{morph_dir} 中没有 png 帧")

    print("run_dir:", run_dir)
    print("img0:", img0_path)
    print("img1:", img1_path)
    print("num frames:", len(morph_paths))
    print("device:", device)

    vae = load_vae(
        diffmorpher_dir=Path(args.diffmorpher_dir),
        ckpt_name=args.ckpt_name,
        cfg_name=args.cfg_name,
        device=device,
    )

    transform = get_preprocess_512()

    v0 = encode_image_to_vae_latent_vector(vae, img0_path, transform, device=device)
    v1 = encode_image_to_vae_latent_vector(vae, img1_path, transform, device=device)

    vs = []
    for p in morph_paths:
        vs.append(encode_image_to_vae_latent_vector(vae, p, transform, device=device))

    X = np.stack([v0, v1] + vs, axis=0)  # [N, D]
    coords, _, _ = pca_2d(X)  # [N, 2]
    coords3, _, _ = pca_3d(X)  # [N, 3]

    c_img0 = coords[0]
    c_img1 = coords[1]
    coords_gen = coords[2:]

    # 输出
    out_img = run_dir / "latent_space_pca.png"
    out_img3d = run_dir / "latent_space_pca_3d.png"
    out_txt = run_dir / "latent_space_pca_points.txt"
    out_npy = run_dir / "latent_space_pca_coords.npy"

    labels = ["img0", "img1"] + [p.stem for p in morph_paths]
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("label\tx_pc1\tx_pc2\n")
        for lab, (a, b) in zip(labels, coords):
            f.write(f"{lab}\t{a:.6f}\t{b:.6f}\n")

    np.save(out_npy, coords)

    # 画图：原图黑色，morph 帧蓝色（按序号逐渐变深）
    n = coords_gen.shape[0]
    frame_alphas = np.linspace(0, 1, n)

    plt.figure(figsize=(7.5, 6.2), dpi=160)
    plt.scatter([c_img0[0]], [c_img0[1]], c="black", s=70, label="img0")
    plt.scatter([c_img1[0]], [c_img1[1]], c="black", s=70, label="img1")
    plt.scatter(coords_gen[:, 0], coords_gen[:, 1], c=frame_alphas, cmap="Blues", s=35, label="morph frames")
    plt.plot(coords_gen[:, 0], coords_gen[:, 1], color="blue", linewidth=1, alpha=0.35)

    # 标注首尾两帧，避免文字太密
    plt.text(coords_gen[0, 0], coords_gen[0, 1], f" {morph_paths[0].stem}", fontsize=8)
    plt.text(coords_gen[-1, 0], coords_gen[-1, 1], f" {morph_paths[-1].stem}", fontsize=8)

    plt.title(f"Latent-space trajectory (VAE latent PCA)\\n{run_dir.name}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_img)

    # 3D 画图：PCA 前 3 维（静态视角）
    n3 = coords3.shape[0] - 2
    frame_alphas = np.linspace(0, 1, n3)
    coords_gen3 = coords3[2:]
    c_img03 = coords3[0]
    c_img13 = coords3[1]

    fig = plt.figure(figsize=(7.5, 6.2), dpi=160)
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter([c_img03[0]], [c_img03[1]], [c_img03[2]], c="black", s=70, label="img0")
    ax.scatter([c_img13[0]], [c_img13[1]], [c_img13[2]], c="black", s=70, label="img1")

    sc = ax.scatter(
        coords_gen3[:, 0],
        coords_gen3[:, 1],
        coords_gen3[:, 2],
        c=frame_alphas,
        cmap="Blues",
        s=25,
        label="morph frames",
    )
    ax.plot(coords_gen3[:, 0], coords_gen3[:, 1], coords_gen3[:, 2], color="blue", linewidth=1, alpha=0.35)

    ax.set_title(f"Latent-space trajectory (VAE latent PCA 3D)\\n{run_dir.name}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_img3d)

    # plotly 交互式导出（如果可用）
    try_make_plotly_3d(
        run_dir=run_dir,
        coords3=coords3,
        morph_paths=morph_paths,
        frame_alphas=frame_alphas,
    )

    print("saved:", out_img)
    print("saved:", out_img3d)
    print("saved:", out_txt)
    print("saved:", out_npy)


if __name__ == "__main__":
    main()

