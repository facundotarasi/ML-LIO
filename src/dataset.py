#!/usr/bin/env python3.8

import data_mods as mod
import argparse

"""
En este codigo vamos a leer las densidades de fitting y las 
energias XC y armamos el DATASET
"""

# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument("-i","--inputs", help="Input file path", action="store")
args = parser.parse_args()

# Checkeamos la correcta ejecucion del programa
if args.inputs == None:
    print("Ejecuta el codigo: ./dataset.py -i path_to_input")
    exit(-1)

# Leemos el archivo de input
inputs_pred = mod.read_input(args.inputs)

# Cargamos los datos y generamos el predictor
pred = mod.Predictor(inputs=inputs_pred)
pred.setup()
