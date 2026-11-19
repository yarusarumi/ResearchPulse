"""Start the desktop launcher without a console or browser for Task Scheduler."""

from pathlib import Path
import os
import runpy
import sys

app = Path(__file__).resolve().parents[1] / "app"
os.environ["PYTHONPATH"] = str(app)
sys.path.insert(0, str(app))
sys.argv.append("--background")
runpy.run_module("neuroei.desktop", run_name="__main__")
