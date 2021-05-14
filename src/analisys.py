#!/usr/bin/env python3.8

"""
Este codigo lee un dataset.pickle y usa su contenido
para hacer un analisis de los datos
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np

def load_data(name):
    try:
        with open(name,"rb") as f:
                data = pickle.load(f)
    except:
        print("No se pudo leer el", end=" ")
        print("archivo", name)
        exit(-1)
    return data
        
def plot_energy(datas,inputs):
    for name in datas:
        E = []
        data = datas[name]
        for mol in data:
            val = mol["T"][0] * 627.5 # Kcal/mol
            E.append(val)
    
        plt.hist(E,bins=100,label=name)
        plt.legend()
    plt.title("Distribucion de Exc")
    plt.xlabel("Energy [Kcal/mol]")
    plt.ylabel("Frecuency")
    path = inputs["path"] + "distribucion_Exc.png"
    plt.savefig(path)
    plt.close()

def plot_density(datas,inputs,type_at):
    if type_at == "H": 
        code = 4
    else: 
        code = 13

    col = 2
    row = int(code / 2) + int(code % 2)

    figure = plt.figure(figsize=(10,10))
    figure.suptitle("Dist. Dens. " + type_at)
    for name in datas:
        # Extramos los datos
        dens = []
        data = datas[name]
        for mol in data:
            for jj in range(len(mol[type_at])):
                dens.append(mol[type_at][jj])
        
        dens = np.array(dens)
        dens = dens.reshape((-1,code))

        # Graficamos
        for jj in range(code):
            figure.add_subplot(row,col,jj+1)
            label= "coef " + str(jj) + " " + name
            hist, bins = np.histogram(dens[:,jj],100)
            hist = hist / max(hist)
            plt.plot(bins[:-1],hist,label=label)
            #plt.hist(dens[:,jj],bins=100,label=label)
            plt.xlabel("Densidad")
            plt.ylabel("Frequency")
            plt.legend()
    path = inputs["path"] + "dist_dens_" + type_at
    path += ".png"
    plt.savefig(path)
    plt.show()
    plt.close()
        

    

inputs = {
    "code_name": ["etano","metano"],
    "path": "/home/gonzalo/Calculos/Datasets/analisys/",
}

# Cargamos todos los Datasets
datas = {}
for code in inputs["code_name"]:
    datas[code] = load_data(inputs["path"]+"dataset_"+code+".pickle")

# Graficamos una distribucion de la energia
plot_energy(datas,inputs)

# Graficamos una distribucion de las densidades de fitting
plot_density(datas,inputs,"C")