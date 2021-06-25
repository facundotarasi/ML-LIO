import time
import numpy as np
import pickle
import yaml

# Esto lee el archivo yaml de input
def read_input(files):
    try:
        with open(files, 'r') as f:
            inp = yaml.load(f,Loader=yaml.FullLoader)
    except:
        print("El archivo " + files + " no existe")
        exit(-1)
    return inp

# Esto genera el dataset con el fingerprint
class Predictor:
    def __init__(self, inputs=None):
        self.path_P = inputs["path_P"]
        self.path_C = inputs["path_C"]
        self.save_pos = inputs["save_pos"]
        self.ndata = inputs["ndata"]
        self.path_results = inputs["path_results"]

    def setup(self):
        # Leemos Coordenadas, Pfit, Exc, Nuc, type, how_many
        data = self._read_files()

        # Con esto simetrizamos las funciones p y d
        # de la densidad de fitting, por lo tanto la dimension se achica
        data = self._symmetrize(data)

        # Separamos en tipo de atomos -> H, C, N, O
        data = self._separate(data)

        # Guardamos el dataset
        self._save_dataset(data)

    def _read_files(self):
        """
        Este archivo tiene que leer todo, Exc, Pfit, coord.
        en que nucleo estan centradas las bases y de q tipo son (s,p,d)
        """
        data = []
        print("Leyendo los archivos...",end=" ")
        init = time.time()

        for ii in range(self.ndata):
            # Leo las coordenadas y el numero atomico
            file_name = self.path_C + str(ii+1) + ".xyz"
            atomic_number, positions, how_many = self._read_xyz(file_name)

            # Leo Densidades, Energia XC, type gaussians y Nuc
            file_name = self.path_P + str(ii+1) + ".dat"
            Exc, Pmat_fit, gtype, Nuc = self._read_rhoE(file_name)

            # Guardo los datos en un solo diccionario
            single_data = {
                "targets": Exc,
                "atomic_number": atomic_number,
                "how_many": how_many,
                "Pmat_fit": Pmat_fit,
                "gtype": gtype, 
                "Nuc": Nuc,
            }

            if len(positions) != 0:
                single_data["positions"] = positions
            
            # Guardo los datos de una molecula
            data.append(single_data)
        
        fin = time.time()
        print(str(np.round(fin-init,2))+" s.")

        return data
    
    def _read_xyz(self,name):
        atomic_number = []
        positions = []
        save = self.save_pos
        how_many = [0, 0, 0, 0]
        try:
            with open(name) as f:
                for line in f:
                    field = line.split()
                    if len(field) == 4:
                        at = int(field[0])
                        x = float(field[1])
                        y = float(field[2])
                        z = float(field[3])
                        atomic_number.append(at)
                        if at == 1:
                            how_many[0] += 1
                        elif at == 6:
                            how_many[1] += 1
                        elif at == 7:
                            how_many[2] += 1
                        elif at == 8:
                            how_many[3] += 1
                        else:
                            print("Solo se acepta CHON por el momento")
                            exit(-1)

                        if save:
                            positions.append([x,y,z])

        except:
            print("El archivo " + name + " no existe")
            exit(-1) 
        
        return atomic_number, positions, how_many

    def _read_rhoE(self,name):
        """
        ? El archivo viene organizado de la siguiente manera:
        ? E1, E2, Exc, En, Etot, M, Md
        ? nshelld: s, p, d, -, -,
        ? Nucd
        ? Pmat, con n_M elementos
        ? Pmat_fitt, con n_Md elementos
        """
        try:
            Exc, Pmat_fit = [], []
            gtype, Nuc = [], []
            Md = -1
            with open(name) as f:
                nlines = 0
                for line in f:
                    nlines += 1
                    field = line.split()

                    # Leo la 1ra linea: saco Exc
                    if len(field) == 7:
                        Exc.append(float(field[2]))
                        Md = int(field[-1])

                    # Leo la 2da linea: tipos de gaussianas
                    if len(field) == 5:
                        for ii in range(3):
                            gtype.append(int(field[ii]))

                    if len(field) == Md:
                        # Leo la 3ra linea: Nuc
                        if nlines == 3:
                            for ii in range(Md):
                                Nuc.append(int(field[ii]))
                        # Leo la 5ta linea: Pmat_fit
                        elif nlines == 5:
                            for ii in range(Md):
                                Pmat_fit.append(float(field[ii]))
        except:
            print("El archivo " + name + " no existe")
            exit(-1) 

        return Exc, Pmat_fit, gtype, Nuc

    def _symmetrize(self,data):
        # La densidad de fitting esta en orden
        # primero estan todas las s ( type[0] )
        # luego estan todas las p ( type[1] )
        # luego estan todas las d ( type[2] )

        data_symm = []
        for ii in range(len(data)):
            ns   = data[ii]["gtype"][0]
            np   = data[ii]["gtype"][1]
            nd   = data[ii]["gtype"][2]

            Exc  = data[ii]["targets"]
            Pmat = data[ii]["Pmat_fit"]
            Pmat_sym, Nuc_sym = [], []

            # Ponemos las s
            for jj in range(0,ns):
                Nuc_sym.append(data[ii]["Nuc"][jj])
                Pmat_sym.append(Pmat[jj])

            # Simetrizamos las p
            for jj in range(ns,ns+np,3):
                temp  = Pmat[jj+0]**2
                temp += Pmat[jj+1]**2
                temp += Pmat[jj+2]**2
                Nuc_sym.append(data[ii]["Nuc"][jj])
                Pmat_sym.append(temp)
            
            # Simetrizamos las d
            for jj in range(ns+np,ns+np+nd,6):
                temp  = Pmat[jj+0]**2
                temp += Pmat[jj+1]**2
                temp += Pmat[jj+2]**2
                temp += Pmat[jj+3]**2
                temp += Pmat[jj+4]**2
                temp += Pmat[jj+5]**2
                Nuc_sym.append(data[ii]["Nuc"][jj])
                Pmat_sym.append(temp)

            
            data_symm.append({
                "Targets": Exc,
                "Pmat_fit": Pmat_sym,
                "Nuc": Nuc_sym,
                "Atomic_number": data[ii]["atomic_number"],
                "How_many": data[ii]["how_many"],
            })

        return data_symm
            
    def _separate(self,data):
        #* Aclaracion: si usamos la base DZVP para el fitting
        #* H = (4s, 0p, 0d)
        #* C = (7s, 3p, 3d)
        #* N = (7s, 3p, 3d)
        #* O = (7s, 3p, 3d)
        data_sep = []

        for mol in data:
            at_no = mol["Atomic_number"]
            Nucd  = mol["Nuc"]
            Pmat  = mol["Pmat_fit"]
            Exc   = mol["Targets"]
            fH, fC, fN, fO = [], [], [], []

            # Separamos la Pfit en cada atom type
            for jj in range(len(Pmat)):
                atom_idx = Nucd[jj] - 1
                atom_type = at_no[atom_idx]
                if atom_type == 1:
                    fH.append(Pmat[jj])
                elif atom_type == 6:
                    fC.append(Pmat[jj])
                elif atom_type == 7:
                    fN.append(Pmat[jj])
                elif atom_type == 8:
                    fO.append(Pmat[jj])
                else:
                    print("El atom type " + str(atom_type),end=" ")
                    print("No es posible")
                    exit(-1)

            data_sep.append({
                "T": Exc,
                "H": fH,
                "C": fC,
                "N": fN,
                "O": fO,
                "how_many": mol["How_many"],
            })
        
        return data_sep

    def _save_dataset(self,data):
        init = time.time()
        path = self.path_results + "dataset_Pfit.pickle"
        try:
            with open(path,"wb") as f:
                pickle.dump(data,
                            f,pickle.HIGHEST_PROTOCOL)
        except:
            print("No se pudor escribir el",end=" ")
            print("Dataset AEV")
            exit(-1)
        print("Escritura",str(np.round(time.time()-init,2))+" s.")