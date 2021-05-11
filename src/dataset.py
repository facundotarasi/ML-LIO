#!/usr/bin/env python3.8

import data_mods as mod

"""
En este codigo vamos a leer las densidades de fitting y las 
energias XC y armamos el DATASET
"""

#! Por el momento no usa los AEVs, solo P fit
inputs_pred = {
    # Path donde estan todos los datos
    "path_P": "../test/data_point/data_point_", 
    "path_C": "../test/coordinates/frames_",
    "file_C": "diazirine.xyz",
    "save_pos": False,
    "ndata": 20000,
}

# Cargamos los datos y generamos el predictor
pred = mod.Predictor(inputs=inputs_pred)
pred.setup()
