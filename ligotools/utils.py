import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import filtfilt, butter
from scipy.io.wavfile import write as wavwrite


def whiten(strain, psd_interp, dt):
    """
    Whiten a strain time series using a (one-sided) interpolated PSD.

    Parameters
    ----------
    strain : 1D np.ndarray
        Time-domain strain.
    psd_interp : callable
        Interpolating function Pxx(f) that returns the (one-sided) PSD.
    dt : float
        Sample spacing (1/fs).

    Returns
    -------
    np.ndarray
        Whitened time series (same length as input).
    """
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    hf = np.fft.rfft(strain)

    # One-sided PSD normalization (dt/2) matches LOSC tutorial convention
    white_hf = hf / np.sqrt(psd_interp(freqs) / (dt / 2.0))
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht

def write_wavfile(basename, fs, data, outdir="audio"):
    """
    Write a WAV file to the given directory.
    Ensures directory exists and saves as basename.wav.
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    fname = os.path.join(outdir, basename + ".wav")

    # Normalize to 16-bit range
    scaled = np.int16(data / np.max(np.abs(data)) * 32767)

    wavwrite(fname, fs, scaled)
    return fname


def reqshift(data, fshift=400.0, sample_rate=4096.0):
    """
    Shift a time series by +fshift Hz via FFT bin roll.
    """
    N = len(data)
    hf = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(N, d=1.0 / sample_rate)
    df = freqs[1] - freqs[0]
    ishift = int(np.round(fshift / df))

    hfs = np.roll(hf, ishift)
    if ishift > 0:
        hfs[:ishift] = 0.0
    elif ishift < 0:
        hfs[ishift:] = 0.0
    shifted = np.fft.irfft(hfs, n=N)
    return shifted

def plot_matched_filter_analysis(
    dets, dt, strain_L1, strain_H1,
    datafreq, template_fft,
    fs, NFFT, psd_window, NOVL,
    dwindow, df, time, template,
    ab, bb, normalization,
    strain_H1_whitenbp, strain_L1_whitenbp,
    tevent, eventname, plottype="png",
    outdir="figures"
):
    """
    Minimal plotting routine that:
      - computes complex SNR(t) for each detector with the provided template,
      - saves three plots per detector (match time, SNR(t), whitened strain),
      - returns a real-valued template projection aligned to the peak.
    This matches the usage from the notebook cell that begins
    '# -- To calculate the PSD of the data, choose an overlap and a window ...'.
    """
    import matplotlib.mlab as mlab  # matches original tutorial use
    os.makedirs(outdir, exist_ok=True)

    # Pack detector data the way the tutorial expects
    det_map = {
        "H1": np.asarray(strain_H1, dtype=float),
        "L1": np.asarray(strain_L1, dtype=float),
    }
    whitebp_map = {
        "H1": np.asarray(strain_H1_whitenbp, dtype=float),
        "L1": np.asarray(strain_L1_whitenbp, dtype=float),
    }

    template_time = np.fft.ifft(template_fft * fs).real  # inverse-FFT of template
    out_template_for_csv = None

    for det in dets:
        strain = det_map[det]

        # PSD for this detector (to compute matched filter)
        Pxx, freqs = mlab.psd(
            strain, Fs=fs, NFFT=NFFT, window=psd_window, noverlap=NOVL
        )
        psd = np.interp(np.abs(datafreq), freqs, Pxx)
        psd[psd == 0] = np.inf

        # Matched filter SNR(t)
        hf = np.fft.fft(strain * dwindow)
        snr_fft = hf * np.conj(template_fft) / psd
        snr = 4 * np.fft.ifft(snr_fft) * df
        snr = snr.real

        # Find peak
        imax = np.argmax(np.abs(snr))
        tmax = time[imax]
        snrmax = np.max(np.abs(snr))

        # Construct a (real) template projection around the event for CSV step
        # (use the same time grid as the data)
        template_match = np.fft.ifft(template_fft * np.exp(2j * np.pi * datafreq * (time[imax] - time[0]))).real

        print(
            f"For detector {det}, maximum at {tmax:.4f} with SNR = {snrmax:.1f}"
        )

        # Save plots
        # 1) Match time
        plt.figure()
        plt.plot(time - tevent, snr, lw=1)
        plt.axvline(tmax - tevent, ls="--", lw=1)
        plt.xlabel(f"time (s) since {tevent}")
        plt.ylabel("SNR")
        plt.title(f"{det} matched-filter SNR — {eventname}")
        plt.savefig(os.path.join(outdir, f"{eventname}_{det}_matchtime.{plottype}"))
        plt.close()

        # 2) SNR only (clean)
        plt.figure()
        plt.plot(time - tevent, snr, lw=1)
        plt.xlabel(f"time (s) since {tevent}")
        plt.ylabel("SNR")
        plt.title(f"{det} SNR — {eventname}")
        plt.savefig(os.path.join(outdir, f"{eventname}_{det}_SNR.{plottype}"))
        plt.close()

        # 3) whitened bandpassed strain for context
        plt.figure()
        plt.plot(time - tevent, whitebp_map[det], lw=0.8)
        plt.xlabel(f"time (s) since {tevent}")
        plt.ylabel("whitened strain (bp)")
        plt.title(f"{det} whitened strain — {eventname}")
        plt.savefig(os.path.join(outdir, f"{eventname}_{det}_matchfreq.{plottype}"))
        plt.close()

        # Return a single template series for CSV export (both dets same length)
        out_template_for_csv = template_match

    return out_template_for_csv
