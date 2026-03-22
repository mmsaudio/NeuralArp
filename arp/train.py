from arp.dataset import GuitarDataset, GuitaDataModule
from arp.model import StreamableModel,KMeanCodebookInitCallback
import pytorch_lightning as pl



wav_file = "./data/guitar_dataset.wav"
segment_length = 32270
batch_size = 128

data_module = GuitaDataModule(
    wav_files = wav_file, 
    note_length = segment_length,
    segment_length = segment_length,
    threshold_on_factor=1e-2,
    threshold_off_factor=1e-4,
    display=False,
    batch_size=batch_size,
    train_val_ratio=0.95,
)

model = StreamableModel(
    lr=1e-4,
    padding='same',)

trainer = pl.Trainer(
    max_epochs=100,
    log_every_n_steps=2,
    accelerator="auto",  # GPU if available
    precision='16-mixed',
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
