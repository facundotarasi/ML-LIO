#!/usr/bin/env python3.8

"""
Este codigo lee un dataset.pickle y usa su contenido
para hacer un analisis de los datos
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np


def read_indices(name):
    numbers = []
    try:
        with open(name, 'r') as file:
            for linea in file:
                numbers.append(int(linea))
    except:
        print("No se pudo leer el",end=" ")
        print("archivo " + name)
        exit(-1)
        
    return numbers

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
    plt.close()
        
def plot_molecules(datas,inputs):
    code = inputs["code_name"][0]
    data = datas[code]
    ndat = len(data)
    path = inputs["path"] + "index_train_val.txt"
    indices = read_indices(path)

    # Aqui la cantidad totales de moleculas y de train_val
    eje_x = ["etano", "etano_train", "propano", "propano_train"]
    eje_y = [ndat//2]
    Ene = []
    eta_train, pro_train = 0, 0
    for jj in range(len(indices)):
        idx = indices[jj]
        how_many = data[idx]["how_many"]
        Ene.append(data[idx]["T"][0]*627.5)
        if how_many[1] == 2:
            eta_train += 1
        elif how_many[1] == 3:
            pro_train += 1
        else:
            print("Error solo tiene q haber etano o propano")
            exit(-1)
    eje_y.append(eta_train)
    eje_y.append(ndat//2)
    eje_y.append(pro_train)
    plt.bar(eje_x,eje_y)
    plt.ylabel("# moleculas")
    plt.xlabel("molecula")
    path = inputs["path"] + "distribution_molecules.png"
    plt.savefig(path)
    plt.close()

    # Graficamos la distribucion de energia
    hist, bins = np.histogram(Ene,100)
    plt.title("Energia XC Etano y Propano")
    plt.plot(bins[:-1],hist,label="datos de Train")
    plt.xlabel("Energy [Kcal/mol]")
    plt.ylabel("# molecules")
    plt.legend()
    path = inputs["path"] + "Energia_distribucion.png"
    plt.savefig(path)
    plt.close()
    


inputs = {
    "code_name": ["etano_propano"],
    "path": "/home/gonzalo/Calculos/Machine_learning/etano_propano_solo/analysis/",
}

# Cargamos todos los Datasets
datas = {}
for code in inputs["code_name"]:
    datas[code] = load_data(inputs["path"]+"dataset_"+code+".pickle")

# Cuando entrenamos con varias moleculas
plot_molecules(datas,inputs)
exit(-1)

# Graficamos una distribucion de la energia
plot_energy(datas,inputs)

# Graficamos una distribucion de las densidades de fitting
plot_density(datas,inputs,"C")