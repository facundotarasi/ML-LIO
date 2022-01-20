#!/usr/bin/env python3.8

import ext_mods as mods
import argparse 

"""
Este script agrupa los datos de moléculas cuyos índices corresponden
a los dados al programa. Los separa en subgrupos de training, 
validation, y test.
"""

# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inputs", help= "Input file path", action= "store")
args = parser.parse_args()

# Evaluamos la correcta ejecucion del programa
if args.inputs == None:
    print("Ejecuta el codigo ./extract.py -i path_to_input")
    exit(-1)

# Leemos el archivo de input
inputs_pred = mods.read_input(args.inputs)

# Cargamos los datos y generamos el predictor
pred = mods.Predictor(inputs = inputs_pred)
pred.setup()
