from arp.dataset import GuitarDataset, GuitaDataModule
from arp.model import StreamableModel,KMeanCodebookInitCallback
import pytorch_lightning as pl



wav_file = "/mnt/c/Users/remi/Desktop/guitare2.wav"
segment_length = 32270
batch_size = 4

dataset = GuitarDataset(
    wav_files=wav_file,
    segment_length=segment_length,
    note_length=44410,
)
data_module = GuitaDataModule(dataset, batch_size=batch_size)

model = StreamableModel(
    lr=1e-4,
    padding='same',)

trainer = pl.Trainer(
    max_epochs=100,
    log_every_n_steps=2,
    accelerator="auto",  # GPU if available
    precision='32-true',
    logger=pl.loggers.CSVLogger("."),
    # logger=pl.loggers.TensorBoardLogger("lightning_logs", name="soundstream"),
    callbacks=[
        pl.callbacks.ModelCheckpoint(save_last=True, every_n_train_steps=50000),
        KMeanCodebookInitCallback(),
    ],
)

trainer.fit(
    model, datamodule=data_module
)
