"""Test setup: import src.* submodules without running the heavy src/__init__ chain (ase, hydra, openbabel...)."""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stub(name: str, path: Path) -> None:
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module


_stub("src", ROOT / "src")
_stub("src.utils", ROOT / "src" / "utils")
_stub("src.pipeline_blocks", ROOT / "src" / "pipeline_blocks")
_stub("src.scripts", ROOT / "src" / "scripts")
