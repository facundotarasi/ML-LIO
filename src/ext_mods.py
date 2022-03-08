import time
import numpy as np
import pickle 
import yaml 
import torch 
import random 

# Esto lee el archivo yaml de input
def read_input(files):
    try:
        with open(files, 'r') as f:
            inp = yaml.load(f, Loader = yaml.FullLoader)
    except:
        print("El archivo {} no existe".format(files))
        exit(-1)
    return inp 

# Esto genera el Dataset 
class Predictor:
    def __init__(self, inputs = None):
        self.path_data = inputs["path_data"]
        self.first_index = inputs["first_index"]
        self.last_index = inputs["last_index"]
        self.path_results = inputs["path_results"]
        self.grid_size = inputs["grid_size"]
        self.coeff = inputs["coeff"]
        self.comps = inputs["comps"]
        self.val_frac = inputs["val_frac"]
        self.test_frac = inputs["test_frac"]

    def setup(self):
        # Si estamos usando como representaciones densidades y gradientes, 
        # las leemos las densidades
        if self.grid_size != False: data = self._read_files_dens()

        # Si se están usando coeficientes y proyecciones, entonces los leemos:
        if self.coeff: data = self._read_files_coeff()

        # Informamos la cantidad de datos del DataSet
        length = len(data)
        print("Numero de datos contenidos: {}".format(str(length)))

        # Dividimos los datos en training, validation y test
        train_data, val_data, test_data = self._split_data(data)

        # Guardamos los DataSets 
        for key in [train_data, val_data, test_data]:
            self._save_dataset(key, length)


    def _read_files_dens(self):
        """
        Este archivo lee los datos en los archivos de input, cuando se trabaja con densidades 
        discretizadas y sus gradientes
        """

        init = time.time()
        data = []
        print("Leyendo los archivos...", end = ' ')

        for key in self.comps:
            for ii in range(self.first_index, self.last_index + 1):

                file_name = self.path_data + key + "_" + str(ii) + ".dat"
                 
                """
                Si se está trabajado con densidades y gradientes discretizados,
                el archivo viene organizado de la siguiente manera:
                * Encabezado (se ignora, son las primeras 14 lineas)
                * Numero de iteracion
                * E1, KinE, Exc, En, Etot, M, Md
                * nshelld: s, p, d, -, -,
                * Iz
                * Densidades
                * Gradientes
                """

                try:
                    with open(file_name) as f:

                        # Inicializamos las listas donde se va a guardar la información de cada
                        # molecula.
                        nlines = 0
                        KinE, Dens, Hw = [], [], []
                        Gradx, Grady, Gradz, Nuc = [], [], [], []
                        itcheck, kincheck, nuccheck = -1, -1, -1
                        for line in f:
                            nlines += 1
                            if nlines < 15:
                                continue 
                            else:
                                field = line.split()

                                # Si es una nueva iteracion, guardamos la informacion anterior e
                                # inicializamos listas nuevas
                                if field[0] == "Iteration":
                                    if len(KinE) == 0:
                                        itcheck = nlines
                                        continue 
                                    else:
                                        size = self.grid_size 
                                        data.append({
                                            "T": torch.tensor(KinE),
                                            "Atno": torch.tensor(Nuc), 
                                            "How_many": torch.tensor(Hw), 
                                            "Dens": torch.tensor(Dens).view(torch.tensor(Nuc).shape[0], size, size, size), 
                                            "Gradx": torch.tensor(Gradx).view(torch.tensor(Nuc).shape[0], size, size, size), 
                                            "Grady": torch.tensor(Grady).view(torch.tensor(Nuc).shape[0], size, size, size), 
                                            "Gradz": torch.tensor(Gradz).view(torch.tensor(Nuc).shape[0], size, size, size),
                                        })
                        
                                    KinE, Dens, Hw = [], [], []
                                    Gradx, Grady, Gradz, Nuc = [], [], [], []
                                    itcheck = nlines
                                
                                # Extraemos el valor de KinE para la iteracion
                                elif len(field) == 7 and (itcheck + 1 == nlines):
                                    KinE.append(float(field[1]))
                                    kincheck = nlines 
                                
                                # Extraemos los numeros atomicos y cuantos nucleos hay
                                elif kincheck + 2 == nlines:
                                    for jj in range(len(field)):
                                        Nuc.append(int(field[jj]))
                                    # El orden en Hw es [H, C, N, O]
                                    Hw = [0, 0, 0, 0]
                                    for nuclei in Nuc:
                                        if nuclei == 1:
                                            Hw[0] += 1
                                        elif nuclei == 6:
                                            Hw[1] += 1
                                        elif nuclei == 7:
                                            Hw[2] += 1
                                        elif nuclei == 8:
                                            Hw[3] += 1
                                        else:
                                            print("Solo se acepta C, H, O, N por el momento")
                                            exit(-1)
                                    nuccheck = nlines 

                                # Extraemos las densidades electronicas
                                elif nuccheck < nlines < nuccheck + len(Nuc) + 1:
                                    dens_at = []
                                    for jj in field:
                                        dens_at.append(float(jj))
                                    Dens.append(dens_at)

                                # Extraemos la derivada de la densidad respecto de x
                                elif nuccheck + len(Nuc) < nlines < nuccheck + 2 * len(Nuc) + 1:
                                    gradx_at = []
                                    for jj in field:
                                        gradx_at.append(float(jj))
                                    Gradx.append(gradx_at)

                                # Extraemos la derivada de la densidad respecto de y
                                elif nuccheck + 2 * len(Nuc) < nlines < nuccheck + 3 * len(Nuc) + 1:
                                    grady_at = []
                                    for jj in field:
                                        grady_at.append(float(jj))
                                    Grady.append(grady_at)
                                
                                # Extraemos la derivada de la densidad respecto de z
                                elif nuccheck + 3 * len(Nuc) < nlines < nuccheck + 4 * len(Nuc) + 1:
                                    gradz_at = []
                                    for jj in field:
                                        gradz_at.append(float(jj))
                                    Gradz.append(gradz_at)

                    size = self.grid_size 
                    data.append({
                        "T": torch.tensor(KinE),
                        "Atno": torch.tensor(Nuc), 
                        "How_many": torch.tensor(Hw), 
                        "Dens": torch.tensor(Dens).view(torch.tensor(Nuc).shape[0], size, size, size), 
                        "Gradx": torch.tensor(Gradx).view(torch.tensor(Nuc).shape[0], size, size, size), 
                        "Grady": torch.tensor(Grady).view(torch.tensor(Nuc).shape[0], size, size, size), 
                        "Gradz": torch.tensor(Gradz).view(torch.tensor(Nuc).shape[0], size, size, size),
                    })

                except:
                    print("El archivo " + file_name + " no existe")
                    exit(-1)
        
        print(str(np.round(time.time() - init, 2)) + "s.")
        return data

    def _read_files_coeff(self):
        """
        Este método lee los datos en los archivos de inputs al trabajar con coeficientes de las funciones
        base y sus proyecciones como descriptores.
        """

        init = time.time()
        data = []
        print("Leyendo los archivos...", end = " ")

        for key in self.comps:
            for ii in range(self.first_index, self.last_index + 1):
                file_name = self.path_data + key + "_" + str(ii) + ".dat"

                """
                Si se usan coeficienes y proyecciones de las funciones base, 
                la organización de los archivos de entrada es:
                * Encabezado (se ignora, son las primeras 14 lineas)
                * Numero de iteracion
                * E1, KinE, Exc, En, Etot, M, Md
                * nshelld: s, p, d, -, -,
                * Iz
                * Nucd
                * Coeficientes
                * Proyecciones
                """

                #try:
                with open(file_name, 'r') as f:
                    # Inicializamos las listas nuevas donde vamos a guardar la información
                    # para cada molécula

                    nlines, itnum = 0, 0
                    KinE, coeff, proj, Nuc, Hw, Nucd = [], [], [], [], [], []

                    for line in f:
                        nlines += 1
                        if nlines < 15:
                            continue 
                        else:
                            field = line.split()
                            if field[0] == "Iteration":
                                itnum += 1 
                                if len(KinE) != 0:
                                    data.append({
                                        "T": torch.tensor(KinE),
                                        "Atno": torch.tensor(Nuc),
                                        "Nucd": torch.tensor(Nucd),
                                        "How_many": torch.tensor(Hw),
                                        "Coeff": torch.tensor(coeff),
                                        "Proj": torch.tensor(proj),
                                    })

                                    KinE, coeff, proj, Nuc, Hw, Nucd = [], [], [], [], [], []

                            # Usamos un identificador para el número de líneas. A nlines le restamos
                            # 14 (el encabezado), y le restamos 7 * el número de iteraciones, ya que cada
                            # iteración da siete líneas de input

                            ident = nlines - 14 - (itnum - 1) * 7
                            if ident == 2:
                                # De acá extraemos la energía cinética
                                KinE.append(float(field[1]))
                            
                            if ident == 4: 
                                # Estos corresponden a los números atómicos
                                for ii in field: Nuc.append(int(ii))
                                # Ahora calculamos cuántos núcleos matómicos hay de cada tipo
                                # El orden de How_many es [H, C, N, O]
                                Hw = [0, 0, 0, 0]
                                for jj in Nuc:
                                    if jj == 1:
                                        Hw[0] += 1
                                    if jj == 6:
                                        Hw[1] += 1
                                    if jj == 7:
                                        Hw[2] += 1
                                    if jj == 8:
                                        Hw[3] += 1
                            
                            if ident == 5:
                                # Estos son los Nucd
                                for ii in field: Nucd.append(int(ii))
                            
                            if ident == 6:
                                # Estos son los coeficientes de las bases, af
                                for ii in field: coeff.append(float(ii))
                            
                            if ident == 7:
                                # Estas son las proyecciones sobre las funciones base
                                for ii in field: proj.append(float(ii))
                
                # Cuando se termina de leer el archivo, se guardan los datos del cálculo convergido
                if len(KinE) != 0:
                    data.append({
                        "T": torch.tensor(KinE),
                        "Atno": torch.tensor(Nuc),
                        "Nucd": torch.tensor(Nucd),
                        "How_many": torch.tensor(Hw),
                        "Coeff": torch.tensor(coeff),
                        "Proj": torch.tensor(proj),
                    })

                #except:
                #    print("El archivo {} no existe".format(file_name))
                #    exit(-1)
                
        print(str(np.round(time.time() - init, 2)) + "s.")
        return data 

    def _split_data(self, data):
        init = time.time()
        print("Separando los datos...", end = ' ')
        train_data, val_data, test_data = [], [], []
        
        # Generamos una lista con numeros enteros entre 0 y el total
        # de datos
        rand_ind = list(range(len(data)))

        # La reordenamos aleatoriamiente
        random.shuffle(rand_ind)
 
        n_val = int(len(data) * self.val_frac)
        n_test = int(len(data) * self.test_frac)

        # Generamos los nuevos Datasets, ya repartidos aleatoriamente
        for ii in range(n_val):
            val_data.append(data[rand_ind[ii]])

        for ii in range(n_val, n_val + n_test):
            test_data.append(data[rand_ind[ii]])
        
        for ii in range(n_val + n_test, len(data)):
            train_data.append(data[rand_ind[ii]])
        
        print(str(np.round(time.time() - init, 2)) + " s.") 
        return train_data, val_data, test_data 

    def _save_dataset(self, dataset, length):
        init = time.time()

        if (len(dataset) == int(self.val_frac * length)): 
            name_data = "DataSet de Validacion"
            name_out = "val_data"
        elif (len(dataset) == int(self.test_frac * length)):
            name_data = "DataSet de Test"
            name_out = "test_data"
        else:
            name_data = "DataSet de Entrenamiento"
            name_out = "train_data"

        print("Escribiendo " + name_data + "... ", end="")

        path = self.path_results + name_out + "_" + str(self.first_index) \
            + "_to_" + str(self.last_index) + ".pickle"
        try:
            with open(path, "wb") as f:
                pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
        except:
            print("No se puedo escribir el DataSet")
            exit(-1)
        print(str(np.round(time.time() - init, 2)) + " s.")
