
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import random
torchaudio.set_audio_backend("soundfile")

class GuitarDataset(torch.utils.data.Dataset):
    def __init__(self, 
        wav_files, 
        segment_length,
        note_length,
        threshold_on_factor=1e-2,
        threshold_off_factor=1e-4,
        display=False
    ):
        if isinstance(wav_files, str):
            wav_files = [wav_files]

        notes = []
        for f in wav_files:
            _notes, fs = wav_to_notes(
                f, 
                n_samples=note_length,
                threshold_on_factor=threshold_on_factor, 
                threshold_off_factor=threshold_off_factor, 
                display=display
            )
            notes.append(_notes)

        notes = torch.cat(notes, dim=0)
        print("Total notes = %i"%len(notes))

        self.notes = notes
        self.fs = fs
        self._segment_length = segment_length

    def __getitem__(self, index):
        x = self.notes[index]
        x *= 0.95 / torch.max(x)
        assert x.dim() == 1
        if x.shape[0] < self._segment_length:
            x = F.pad(x, [0, self._segment_length - x.shape[0]], "constant")
        pos = random.randint(0, x.shape[0] - self._segment_length)
        x = x[pos:pos + self._segment_length]
        return x

    def __len__(self):
        return len(self.notes)


def wav_to_notes(wav_file, n_samples=44100, threshold_on_factor = 0.1, threshold_off_factor=0.001, display=False):
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

    min_dist = 21
    threshold_on =threshold_on_factor*flux.max()
    threshold_off = energy.max() *threshold_off_factor
    pooled = F.max_pool1d(
        flux[..., None, None,:],
        kernel_size=min_dist,
        stride=1,
        padding=min_dist // 2
    )[0, 0]
    onset = (flux == pooled) & (flux > threshold_on)

    note_start = time[n_fft//4:-n_fft//4:n_fft//2][onset]
    note_mask = time[::n_fft//2][None] >= note_start[..., None]
    note_mask = note_mask &  (time[::n_fft//2][None] < torch.cat((note_start[1:], time[[-1]]))[..., None])
    note_mask = note_mask & (energy>threshold_off)[None]
    note_end = time[::n_fft//2][(note_mask * torch.arange(energy.shape[0])[None]).max(dim=1)[0]]
    n_notes= len(note_start)

    note_start_idx = torch.floor(note_start * fs).long()
    note_end_idx   = torch.ceil(note_end * fs).long()
    note_idx = torch.arange(n_samples).unsqueeze(0) + note_start_idx.unsqueeze(1)  # [N, N_SAMPLES]
    mask = (note_idx < note_end_idx.unsqueeze(1)) & (note_idx < waveform.shape[0])

    note_idx = torch.clamp(note_idx, max=waveform.shape[0] - 1)
    notes = waveform[note_idx]
    notes = notes * mask

    if display:
        fig, ax = plt.subplots(1,1, figsize=(10,5))
        offset = 0
        ax.plot(time,  waveform.numpy(), color="grey", alpha=0.5)
        for i in range(n_notes):
            ax.plot((torch.arange(n_samples)+note_start_idx[i])/fs, notes[i] + offset)
        fig.show()

    print("Extracted %i notes from wave file %s "%(len(notes), wav_file))

    return notes, fs