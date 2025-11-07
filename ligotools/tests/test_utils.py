import numpy as np
from pathlib import Path
from scipy.io import wavfile as wav

from ligotools.utils import whiten, write_wavfile, reqshift


def test_whiten_length_and_finiteness():
    # synthetic signal
    fs = 4096
    n = 4096
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 100 * t) + 0.1 * np.random.randn(n)

    psd_interp = lambda f: np.ones_like(f)
    y = whiten(x, psd_interp, 1.0 / fs)

    assert y.shape == x.shape
    assert np.isfinite(y).all()
    assert np.var(y) > 0.0  # not collapsed


def test_reqshift_zero_is_identity():
    fs = 2048
    n = 4096
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 50 * t)
    # zero shift should return (almost) the same signal
    y = reqshift(x, fshift=0.0, sample_rate=fs)
    assert np.allclose(y, x, atol=1e-10)


def test_write_wavfile_roundtrip(tmp_path):
    fs = 4096
    n = 4096
    t = np.arange(n) / fs
    x = 0.5 * np.sin(2 * np.pi * 220 * t)

    write_wavfile("test_signal", fs, x, outdir=str(tmp_path))

    out = Path(tmp_path) / "test_signal.wav"
    assert out.exists()

    rfs, rdata = wav.read(out)
    assert rfs == fs
    assert len(rdata) == len(x)