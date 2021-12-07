#!/usr/bin/env python3.8

import modules as mod
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
import argparse

# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument("-i","--inputs", help="Input file path", action="store")
args = parser.parse_args()

# Checkeamos la correcta ejecucion del programa
if args.inputs == None:
    print("Ejecuta el codigo: ./test.py -i path_to_input")
    exit(-1)

# Leemos el archivo de input
inputs = mod.read_input(args.inputs)

# Seteamos la misma seed para todo
pl.seed_everything(inputs["seed"], workers=True)

# Nombre del archivo de datos
inputs["dataset_file"] = inputs["path_dir"] + "dataset_Pfit.pickle"

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

