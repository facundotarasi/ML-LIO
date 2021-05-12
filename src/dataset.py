#!/usr/bin/env python3.8

import data_mods as mod

"""
En este codigo vamos a leer las densidades de fitting y las 
energias XC y armamos el DATASET
"""

#! Por el momento no usa los AEVs, solo P fit
inputs_pred = {
    # Path donde estan todos los datos
    "path_P": "/home/gonzalo/Calculos/Datasets/diazirina/data/data_point_", 
    "path_C": "/home/gonzalo/Calculos/Datasets/diazirina/coordinates/frames_",
    "save_pos": True,
    "ndata": 20000,

    # Folder donde guardar todos los resultados
    "path_results": "/home/gonzalo/Calculos/Datasets/diazirina/"
}

# Cargamos los datos y generamos el predictor
pred = mod.Predictor(inputs=inputs_pred)
pred.setup()
