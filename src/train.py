#!/usr/bin/env python3.8

import modules as mod
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
import argparse

# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument("-i","--inputs", help = "Input file path", action = "store")
args = parser.parse_args()

# Chequeamos la correcta ejecucion del programa
if args.inputs == None:
    print("Ejecuta el codigo: ./train.py -i path_to_input")
    exit(-1)

# Leemos el archivo de input
inputs = mod.read_input(args.inputs)

# Seteamos la misma seed para todo
pl.seed_everything(inputs["seed"], workers=True)

# Resultados
inputs["path_results"] = inputs["path_dir"] + inputs["folder"]

Data = mod.DataModule(inputs)
Data.setup(stage = "fit")

# Instanciamos el Modelo
if inputs["restart"]:
    print("Reiniciando entrenamiento...")
    path = inputs["path_results"]
    path = path + inputs["model_file"]
    try:
        model = mod.Modelo.load_from_checkpoint(
                checkpoint_path=path,config=inputs)
    except:
        print("El modelo no se pudo leer")
        exit(-1)
else:
    model = mod.Modelo(inputs)

# Checkpoint del Modelo
if not inputs["restart"]: checkpoint = ModelCheckpoint(
    dirpath = inputs["path_results"],
    monitor="val_loss",
    filename = "modelo-{val_loss:.5f}",
    save_top_k = 1,
    mode="min"
)

# Early stopping
early_stopping = EarlyStopping(
    monitor = "val_loss",
    patience= 20,
    verbose=True,
    mode="min",
    min_delta = 0.000001
)

# Lr Monitor
lr_monitor = LearningRateMonitor(logging_interval="epoch")

# Entrenamiento
calls = [checkpoint,early_stopping,lr_monitor]
if inputs["restart"]:
    trainer = pl.Trainer(max_epochs=inputs["nepochs"], gpus=inputs["gpu"],callbacks=calls, resume_from_checkpoint = 
    inputs["path_results"] + inputs["model_file"])
else:
    trainer = pl.Trainer(max_epochs=inputs["nepochs"], gpus=inputs["gpu"],callbacks=calls)
trainer.fit(model=model,datamodule=Data)

# Graficamos resultados del train
model.graph_train()
