#!/usr/bin/env python3.8

import modules as mod
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

"""
En este codigo vamos a leer las densidades de fitting y las 
energias XC y armamos el DATASET
"""

# Inputs necesarios para leer la data y generar los
# fingerprints atomicos de cada molecula
# * Los fingerprints en este caso tienen la dens. de fit.
# ! Por el momento solo anda con densidades de SCF convergido
inputs_pred = {
    # Input Data
    "path": "../data_points_all/data_point_",
    "ndata": 20000,
}

# Cargamos los datos y generamos el predictor
pred = mod.Predictor(inputs=inputs_pred)
pred.setup()