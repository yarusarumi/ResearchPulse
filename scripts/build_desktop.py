"""Build and verify a relocatable ResearchPulse package using Windows x64 CPython 3.12."""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ROOT / "deployment" / "desktop"


def install_runtime(target: Path, uv: str | None) -> Path:
    source = Path(sys.base_prefix)
    target.mkdir()
    for name in ("DLLs", "tcl"):
        if (source / name).is_dir():
            shutil.copytree(source / name, target / name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(
        source / "Lib",
        target / "Lib",
        ignore=shutil.ignore_patterns("site-packages", "EXTERNALLY-MANAGED", "__pycache__", "test", "tests"),
    )
    for name in ("python.exe", "pythonw.exe", "python312.dll", "python3.dll", "LICENSE.txt"):
        shutil.copy2(source / name, target)
    for path in source.glob("vcruntime*.dll"):
        shutil.copy2(path, target)

    python = target / "python.exe"
    if uv:
        subprocess.run(
            [uv, "pip", "install", "--python", str(python), "--requirement",
             str(FILES / "requirements-core-win.lock"), "--no-deps", "--only-binary", ":all:", "--link-mode", "copy"],
            check=True,
        )
    else:
        site_packages = target / "Lib" / "site-packages"
        site_packages.mkdir()
        (site_packages / "neuroei-reader-packages.pth").write_text(
            "import os, site, sys; site.addsitedir(os.path.join(sys.prefix, os.pardir, 'reader-packages'))\n",
            encoding="ascii",
        )
    return python


def copy_readers(assets: Path, destination: Path, paperflow_wheel: Path):
    packages = assets / "reader-packages"
    docling = assets / "models" / "docling"
    if not (packages / ".neuroei-ready").is_file() or not (assets / "models" / ".neuroei-ready").is_file():
        raise SystemExit("Install and verify the Windows reader assets first with deployment/desktop/install_readers.py.")
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc", "guardrail_benchmarks")

    def ignore_package_files(directory, names):
        excluded = ignored(directory, names)
        # The desktop calls its bundled Python directly. Generated console
        # launchers are unused and can embed the build machine's paths.
        return excluded | ({"bin"} if Path(directory) == packages else set())

    shutil.copytree(packages, destination / "reader-packages", ignore=ignore_package_files)
    with zipfile.ZipFile(paperflow_wheel) as wheel:
        wheel.extractall(destination / "reader-packages")
    shutil.copytree(docling, destination / "models" / "docling", ignore=shutil.ignore_patterns(".cache"))
    (destination / "models" / ".neuroei-docling-ready").write_text("ready\n", encoding="ascii")


def build(destination: Path, uv: str, layout_only: bool, reader_assets: Path | None, paperflow_wheel: Path | None) -> Path:
    if sys.platform != "win32" or platform.machine() not in ("AMD64", "x86_64") or sys.version_info[:2] != (3, 12):
        raise SystemExit("Use 64-bit CPython 3.12 on Windows to build this package.")
    archive = destination.with_name(destination.name + ".zip")
    pending = archive.with_suffix(".zip.partial")
    if destination.exists() or archive.exists() or pending.exists():
        raise SystemExit(f"Choose a new output path. Existing build preserved: {destination}")
    if reader_assets and (not paperflow_wheel or paperflow_wheel.name != "paperflow-0.1.0-py3-none-any.whl"
                          or not paperflow_wheel.is_file()):
        raise SystemExit("Supply the production-pinned PaperFlow wheel with --paperflow-wheel.")
    if not reader_assets:
        uv = shutil.which(uv)
        if not uv:
            raise SystemExit("The core-only builder requires uv on PATH or --uv with an executable path.")
    destination.mkdir(parents=True)
    print("Preparing portable Python", flush=True)
    python = install_runtime(destination / "runtime", None if reader_assets else uv)
    shutil.copytree(
        ROOT / "src" / "neuroei",
        destination / "app" / "neuroei",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    common = (
        ("launch.cmd", "启动研究台.cmd"),
        ("schedule.cmd", "配置每日检查.cmd"),
        ("schedule.ps1", "tools/schedule.ps1"),
        ("start_background.pyw", "tools/start_background.pyw"),
        ("install_readers.py", "tools/install_readers.py"),
    )
    core_only = (
        ("install-readers.cmd", "安装论文深读组件.cmd"),
        ("requirements-readers-win.lock", "tools/requirements-readers-win.lock"),
        ("licenses/uv-LICENSE-APACHE", "tools/uv-LICENSE-APACHE"),
        ("licenses/uv-LICENSE-MIT", "tools/uv-LICENSE-MIT"),
        ("使用说明.txt", "使用说明.txt"),
    )
    full = (
        ("install-bge.cmd", "安装 BGE-M3.cmd"),
        ("使用说明-完整组件.txt", "使用说明.txt"),
    )
    for source, target in common + (full if reader_assets else core_only):
        path = destination / target
        path.parent.mkdir(exist_ok=True)
        if path.suffix == ".ps1":
            # Windows PowerShell 5 recognizes non-ASCII source with a UTF-8 BOM.
            path.write_text((FILES / source).read_text(encoding="utf-8-sig"), encoding="utf-8-sig")
        else:
            shutil.copy2(FILES / source, path)
    if reader_assets:
        print("Including Windows CPU dependencies and Docling models", flush=True)
        copy_readers(reader_assets, destination, paperflow_wheel)
    else:
        shutil.copy2(uv, destination / "tools" / "uv.exe")

    # Import from the packaged runtime, including native DLLs used by image transcription.
    smoke = (
        "import sys; from pathlib import Path;"
        "sys.path.insert(0, str(Path(sys.argv[1])));"
        "import neuroei.portal, onnxruntime, cv2, PIL;"
        "assert 'CPUExecutionProvider' in onnxruntime.get_available_providers()"
    )
    env = {**os.environ, "PATH": str(Path(os.environ["SystemRoot"]) / "System32")}
    subprocess.run([str(python), "-IB", "-c", smoke, str(destination / "app")], check=True, env=env)
    if reader_assets:
        check = ("import importlib, torch; from paperqa_docling import parse_pdf_to_pages;"
                 "importlib.import_module('deployments.feishu.daily-push-agent.main');"
                 "assert torch.version.cuda is None; assert torch.ones(2).tolist() == [1.0, 1.0]")
        env.update(LITELLM_LOCAL_MODEL_COST_MAP="True", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   PYTHON_DOTENV_DISABLED="1")
        subprocess.run([str(python), "-IB", "-c", check], check=True, env=env)
    if layout_only:
        return destination
    print("Compressing Windows ZIP", flush=True)
    with zipfile.ZipFile(pending, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=2) as output:
        for path in destination.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                output.write(path, path.relative_to(destination))
    pending.rename(archive)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--name", default="ResearchPulse-Desktop-win64-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--reader-assets", type=Path, help="Installed CPU reader profile to include in the package")
    parser.add_argument("--paperflow-wheel", type=Path, help="Wheel built from the production-pinned PaperFlow revision")
    parser.add_argument("--layout-only", action="store_true")
    args = parser.parse_args()
    print(build(args.output_dir.resolve() / args.name, args.uv, args.layout_only, args.reader_assets,
                args.paperflow_wheel), flush=True)


if __name__ == "__main__":
    main()
