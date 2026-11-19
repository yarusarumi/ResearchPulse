"""Start ResearchPulse from the windowed launcher and surface startup errors."""
import ctypes
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))


def main():
    try:
        from neuroei.desktop import main as run_desktop

        run_desktop()
    except SystemExit as exc:
        if not exc.code:
            return 0
        message = "无法启动研究台。请检查本机 8768 端口是否被其他程序占用。"
    except Exception as exc:
        message = f"无法启动研究台（{type(exc).__name__}）。请检查解压目录权限和运行组件。"
    else:
        return 0
    ctypes.windll.user32.MessageBoxW(None, message, "ResearchPulse", 0x10)
    return 1


if __name__ == "__main__":
    sys.exit(main())
