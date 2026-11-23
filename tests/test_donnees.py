import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analyse import charger  # noqa: E402


def test_jeu_complet():
    df = charger()
    assert len(df) == 41188
    assert 0.10 < df["souscrit"].mean() < 0.12


def test_pas_de_fuite():
    src = (Path(__file__).resolve().parents[1] / "src" / "analyse.py").read_text(encoding="utf-8")
    assert '"duration"' in src.split("exclues =")[1].splitlines()[0]
