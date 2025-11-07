import numpy as np
from pathlib import Path

from ligotools import readligo as rl

def test_loaddata_shapes():
    """Test that loaddata returns strain and time arrays of the same length."""
    root = Path(__file__).resolve().parents[2]   # repo root
    data_dir = root / "data"

    h1_files = list(data_dir.glob("H-*.hdf5"))
    assert len(h1_files) > 0, "No H1 file found in data/"

    strain, time, chan_dict = rl.loaddata(str(h1_files[0]), 'H1')

    assert len(strain) == len(time)
    assert len(strain) > 0
    assert np.all(np.diff(time) > 0)   # time must increase


def test_sampling_rate():
    """Test that sampling rate is about 4096 Hz."""
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "data"

    h1_files = list(data_dir.glob("H-*.hdf5"))
    assert len(h1_files) > 0

    strain, time, chan_dict = rl.loaddata(str(h1_files[0]), 'H1')

    dt = np.mean(np.diff(time))
    fs = 1/dt

    assert np.isclose(fs, 4096, atol=1.0)
