import pathlib, sys
sys.path.insert(0, str(pathlib.Path('src').resolve()))
from data.fetch import get_bars, interval_to_seconds

def test_interval_to_seconds():
    assert interval_to_seconds('1d') == 86400
    assert interval_to_seconds('1h') == 3600
