import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

torchaudio.set_audio_backend("soundfile")

def wav_to_notes(wav_file):
    waveform, fs = torchaudio.load(wav_file)
    waveform = waveform.mean(dim=0, keepdim=False)
    # waveform = waveform[int(10*fs): int(20*fs)]
    n = len(waveform)
    n_fft = 1024
    time = torch.arange(n)/fs
    freqs = torch.linspace(0, fs/2,n_fft//2 + 1)

    transform = torchaudio.transforms.Spectrogram(n_fft=n_fft)
    spec = transform(waveform)
    spec_mag = torch.sqrt(spec + 1e-10)
    diff = spec_mag[:, 1:] - spec_mag[:, :-1]
    diff = torch.clamp(diff, min=0)
    flux = diff.sum(dim=0)
    flux /= flux.max()
    energy = (spec.abs() ** 2).sum(dim=0)
    energy = energy / (energy.max() + 1e-8)

    fig, ax = plt.subplots(1,1, figsize=(10,5))
    ax.imshow(spec.log().numpy(), aspect='auto', origin='lower',
            extent=[time[0], time[-1], freqs[0], freqs[-1]])
    # ax.set_yscale("log")
    fig.show()

    min_dist = 21
    threshold = 0.1*flux.max()
    threshold_off = energy.max() *0.001
    pooled = F.max_pool1d(
        flux[..., None, None,:],
        kernel_size=min_dist,
        stride=1,
        padding=min_dist // 2
    )[0, 0]
    onset = (flux == pooled) & (flux > threshold)

    note_start = time[n_fft//4:-n_fft//4:n_fft//2][onset]
    note_mask = time[::n_fft//2][None] >= note_start[..., None]
    note_mask = note_mask &  (time[::n_fft//2][None] < torch.cat((note_start[1:], time[[-1]]))[..., None])
    note_mask = note_mask & (energy>threshold_off)[None]
    note_end = time[::n_fft//2][(note_mask * torch.arange(energy.shape[0])[None]).max(dim=1)[0]]
    n_notes= len(note_start)

    fig, ax = plt.subplots(1,1, figsize=(10,5))
    ax.plot(time,  waveform.numpy()*100, color="black", alpha=0.5)
    ax.plot(note_start, torch.ones(n_notes), "x", color="red")
    ax.plot(note_end, torch.ones(n_notes),"x", color="blue")
    # ax.plot(time[n_fft//4:-n_fft//4:n_fft//2], (flux).numpy(), color="red")
    # ax.plot(time[n_fft//4:-n_fft//4:n_fft//2], (onset).numpy(), color="red")
    # ax.plot(time[::n_fft//2], (energy).numpy(), color="blue")
    # ax.plot(time[::n_fft//2][energy>threshold_off], (energy[energy>threshold_off]).numpy(),".", color="green")
    fig.show()





    N_SAMPLES = (note_end_idx - note_start_idx).max()

    note_start_idx = torch.floor(note_start * fs).long()
    note_end_idx   = torch.ceil(note_end * fs).long()
    note_idx = torch.arange(N_SAMPLES).unsqueeze(0) + note_start_idx.unsqueeze(1)  # [N, N_SAMPLES]
    mask = (note_idx < note_end_idx.unsqueeze(1)) & (note_idx < waveform.shape[0])

    note_idx = torch.clamp(note_idx, max=waveform.shape[0] - 1)
    notes = waveform[note_idx]
    notes = notes * mask

    fig, ax = plt.subplots(1,1, figsize=(10,5))
    offset = 0
    ax.plot(time,  waveform.numpy(), color="grey", alpha=0.5)
    for i in range(n_notes):
        ax.plot((torch.arange(N_SAMPLES)+note_start_idx[i])/fs, notes[i] + offset)

    fig.show()

    return notes, fs