"""Install the pinned offline reading engine and fetch the public model weights."""

import argparse
import os
import site
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BGE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
MODELS = (
    ("BAAI/bge-m3", BGE_REVISION, "bge-m3",
     ("config.json", "config_sentence_transformers.json", "modules.json", "sentence_bert_config.json",
      "1_Pooling/config.json", "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json",
      "sentencepiece.bpe.model", "pytorch_model.bin", "README.md")),
    ("docling-project/docling-layout-heron", "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8",
     "docling/docling-project--docling-layout-heron",
     ("config.json", "preprocessor_config.json", "model.safetensors", "README.md")),
    ("docling-project/docling-models", "fc0f2d45e2218ea24bce5045f58a389aed16dc23",
     "docling/docling-project--docling-models",
     ("config.json", "README.md", "model_artifacts/tableformer/accurate/tm_config.json",
      "model_artifacts/tableformer/accurate/tableformer_accurate.safetensors")),
)


def install_packages(profile: Path, uv: Path) -> Path:
    site_packages = profile / "reader-packages"
    if (site_packages / ".neuroei-ready").is_file():
        return site_packages
    if site_packages.exists():
        raise SystemExit(f"检测到已有的论文深读组件目录，请先检查内容并备份：{site_packages}")
    staged = profile / ("reader-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    lock = Path(__file__).with_name("requirements-readers-win.lock")
    print("正在下载安装 CPU 深读组件，可以关闭后重新运行。", flush=True)
    subprocess.run(
        [str(uv), "--no-config", "pip", "install", "--python", sys.executable,
         "--cache-dir", str(profile / "downloads" / "uv"), "--target", str(staged),
         "--only-binary=:all:", "--no-deps", "--torch-backend", "cpu", "--link-mode", "copy",
         "-r", str(lock)], check=True,
    )
    smoke = (
        "import site, sys;"
        "sys.path.insert(0, sys.argv[1]); site.addsitedir(sys.argv[1]);"
        "import torch, torchvision, onnxruntime;"
        "from sentence_transformers import SentenceTransformer;"
        "from paperqa_docling import parse_pdf_to_pages;"
        "assert torch.version.cuda is None;"
        "assert (torch.ones(2, device='cpu') + 1).tolist() == [2.0, 2.0]"
    )
    subprocess.run([sys.executable, "-IB", "-c", smoke, str(staged)], check=True)
    (staged / ".neuroei-ready").write_text(BGE_REVISION + "\n", encoding="ascii")
    staged.rename(site_packages)
    return site_packages


def download_models(profile: Path, specs):
    from huggingface_hub import snapshot_download

    for repo, revision, folder, files in specs:
        destination = profile / "models" / folder
        print(f"准备模型 {repo}", flush=True)
        snapshot_download(
            repo_id=repo, revision=revision, allow_patterns=files, token=False, local_dir=destination,
        )
        missing = [file for file in files if not (destination / file).is_file()]
        if missing:
            raise RuntimeError(f"模型文件下载不完整：{repo}: {', '.join(missing)}")
    (profile / "models" / ".neuroei-bge-ready").write_text(BGE_REVISION + "\n", encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    profile = Path(__file__).resolve().parents[1] / "data"
    parser.add_argument("--profile", type=Path, default=profile)
    parser.add_argument("--uv", type=Path, default=Path(__file__).with_name("uv.exe"))
    parser.add_argument("--bge-only", action="store_true", help="Download the BGE-M3 weights for the full desktop ZIP")
    parser.add_argument("--dependencies-only", action="store_true", help="Install and test Windows CPU wheels")
    args = parser.parse_args()
    profile = args.profile.resolve()
    profile.mkdir(parents=True, exist_ok=True)
    for key, value in {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "0",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HOME": str(profile / "downloads" / "huggingface"),
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "LITELLM_TELEMETRY": "False",
    }.items():
        os.environ[key] = value

    if args.bge_only:
        download_models(profile, MODELS[:1])
        print("检索模型准备完毕。请重启研究台。", flush=True)
        return
    site_packages = install_packages(profile, args.uv)
    if not args.dependencies_only:
        sys.path.insert(0, str(site_packages))
        site.addsitedir(str(site_packages))
        download_models(profile, MODELS)
        (profile / "models" / ".neuroei-ready").write_text(BGE_REVISION + "\n", encoding="ascii")
    print("论文深读组件准备完毕。请重启研究台。", flush=True)


if __name__ == "__main__":
    main()
