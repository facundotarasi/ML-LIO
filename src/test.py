#!/usr/bin/env python3.8

import modules as mod
import torch
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl

inputs = {
    # Seed usada
    "seed": 123,

    # Directorio de trabajo
    "path_dir": "/home/gonzalo/Calculos/Machine_learning/etaprop_train_but_val/",

    # Datos para entrenar y validar
    "n_train_val": 0,
    "test_size"  : 0.2,
    "batch_size" : 100,
    "nepochs"    : 0,
    "Loss": "L1Loss", # esta es la MAE
    "Optimizer": "Adam",
    "lr": 1e-2,
    "lr_decay": 0.0,

    # Modelo
    "restart": False,
    "model_file": "modelo-val_loss=0.19939.ckpt",

    # Esta variable indica si estoy testeando o en produccion
    "mode": "production",

    #! Esto solo es valido para el train, aqui no se usa
    "val_options": "in",
    "val_path": "/home/",
}

# Nombre del archivo de datos
inputs["dataset_file"] = inputs["path_dir"] + "dataset_but.pickle"

# Resultados
inputs["path_results"] = inputs["path_dir"] + "results/"

Data = mod.DataModule(inputs)
Data.setup(stage="test")

# Guardamos la cantidad de data total: train + val + test
inputs["ndata"] = Data.ndata

# Cargamos el modelo
path = inputs["path_results"] + "/" 
path = path + inputs["model_file"]
try:
    model = mod.Modelo.load_from_checkpoint(checkpoint_path=path,config=inputs)
except:
    print("El modelo no se pudo Leer")
    exit(-1)

# Instanciamos el Trainer
trainer = pl.Trainer(max_epochs=0, gpus=0)
trainer.test(model=model,datamodule=Data)

# Graficamos resultados del test
model.graficar_test(Data.factor_norm)

