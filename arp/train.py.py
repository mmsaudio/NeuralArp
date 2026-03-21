from arp.dataset import GuitarDataset



def train():
    wav_file = "/mnt/c/Users/remi/Desktop/guitare2.wav"

    dataset = GuitarDataset(
        wav_files=wav_file,
        segment_length=4410,
        note_length=44410,
    )

    model = StreamableModel(
        batch_size=32,
        sample_rate=dataset.fs,
        segment_length=32270,
        padding='same',
        dataset='librispeech')
    trainer = pl.Trainer(
        max_epochs=10000,
        log_every_n_steps=2,
        precision='16-mixed',
        logger=pl.loggers.CSVLogger("."),
        # logger=pl.loggers.TensorBoardLogger("lightning_logs", name="soundstream"),
        callbacks=[
            pl.callbacks.ModelCheckpoint(save_last=True, every_n_train_steps=50000),
            KMeanCodebookInitCallback(),
        ],
    )
    trainer.fit(
        model,
    )

    return model
