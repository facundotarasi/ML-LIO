import time
import numpy as np
import pickle

# Esto genera el dataset con el fingerprint
class Predictor:
    def __init__(self, inputs=None):
        self.path_P = inputs["path_P"]
        self.path_C = inputs["path_C"]
        self.save_pos = inputs["save_pos"]
        self.ndata = inputs["ndata"]

    def setup(self):
        # Leemos Coordenadas, Pfit y Exc
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

        # TODO: la modificacion para q utilize varias densidades por xyz
        # TODO: tiene q estar en esta rutina

        for ii in range(self.ndata):
            # Leo las Densidades y las Energias
            file_name = self.path_P + str(ii+1) + ".dat"
            single_data = self._read_one_file(file_name)

            # Leo las coordenadas y el numero atomico
            file_name = self.path_C + str(ii+1) + ".xyz"
            atomic_number, positions = self._read_xyz(file_name)

            # Guardo los datos en un solo diccionario
            single_data["atomic_number"] = atomic_number
            if len(positions) != 0:
                single_data["positions"] = positions
            
            # Leemos las centros de las gaussianas y tipos
            # TODO: ver donde poner esto para leer los centros de las gaussianas
            gtype, Nuc = self._read_Nuc("Nucd_file.dat")
            single_data["type"] = gtype
            single_data["Nuc"]  = Nuc

            # Guardo ambos valores
            data.append(single_data)
        
        fin = time.time()
        print(str(np.round(fin-init,2))+" s.")
        return data
    
    def _read_one_file(self,name):
        """}
        ? El archivo viene organizado de la siguiente manera
        ? E1, E2, Exc, Ehf, M, Md
        ? Pmat, con n_M elementos
        ? Pmat_fitt, con n_Md elementos
        * Solo extraigo Exc y Pmat_fitt
        """
        n1, Md = 6, 110
        try:
            Exc, Pmat_fitt = [], []
            with open(name) as f:
                for line in f:
                    field = line.split()

                    # Con esto leo la 1ra linea (Energias)
                    if (len(field) == n1):
                        Exc.append(float(field[2]))

                    # Con esto leo la 3ra linea (Pmat_fitt)
                    if (len(field) == Md):
                        for ii in range(Md):
                            Pmat_fitt.append(float(field[ii]))
        except:
            print("El archivo " + name + " no existe")
            exit(-1) 

        dicc = {
            "Exc": Exc,
            "Pmat_fit": Pmat_fitt
        }

        return dicc

    def _read_xyz(self,name):
        atomic_number = []
        positions = []
        save = self.save_pos
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
                        if save:
                            positions.append([x,y,z])

        except:
            print("El archivo " + name + " no existe")
            exit(-1) 
        
        return atomic_number, positions
        
    def _read_Nuc(self,name):
        """
        ? El archivo viene organido de la siguiente manera:
        ?  ns, np, nd
        ?  Nucd
        """
        n1, Md = 3, 110
        gtype, Nuc = [], []
        try:
            Exc, Pmat_fitt = [], []
            with open(name) as f:
                for line in f:
                    field = line.split()

                    # Con esto leo la 1ra linea (Energias)
                    if (len(field) == n1):
                        Exc.append(float(field[2]))
                        gtype.append(int(field[0])) # s
                        gtype.append(int(field[1])) # p
                        gtype.append(int(field[2])) # d

                    # Con esto leo la 2da linea con Nucd
                    if (len(field) == Md):
                        for ii in range(Md):
                            Nuc.append(int(field[ii]))
        except:
            print("El archivo " + name + " no existe")
            exit(-1) 

        # gtype contiene la cantidad de s, p y d ( 0, 1, 2 )
        # Nuc contiene a indice atomico le pertenece la gaussiana de la 
        # densidad de fitting
        return gtype, Nuc

    def _symmetrize(self,data):
        # La densidad de fitting esta en orden
        # primero estan todas las s ( type[0] )
        # luego estan todas las p ( type[1] )
        # luego estan todas las d ( type[2] )

        data_symm = []
        for ii in range(len(data)):
            ns   = data[ii]["type"][0]
            np   = data[ii]["type"][1]
            nd   = data[ii]["type"][2]

            Exc  = data[ii]["Exc"]
            Pmat = data[ii]["Pmat_fit"]
            Pmat_sym = []
            Nuc_symm  = []
            new_ns, new_np, new_nd = 0, 0, 0

            # Ponemos las s
            for jj in range(0,ns):
                new_ns += 1
                Nuc_symm.append(data[ii]["Nuc"][jj])
                Pmat_sym.append(Pmat[jj])

            # Simetrizamos las p
            for jj in range(ns,ns+np,3):
                temp  = Pmat[jj+0]**2
                temp += Pmat[jj+1]**2
                temp += Pmat[jj+2]**2
                new_np += 1
                Nuc_symm.append(data[ii]["Nuc"][jj])
                Pmat_sym.append(temp)
            
            # Simetrizamos las d
            for jj in range(ns+np,ns+np+nd,6):
                temp  = Pmat[jj+0]**2
                temp += Pmat[jj+1]**2
                temp += Pmat[jj+2]**2
                temp += Pmat[jj+3]**2
                temp += Pmat[jj+4]**2
                temp += Pmat[jj+5]**2
                new_nd += 1
                Nuc_symm.append(data[ii]["Nuc"][jj])
                Pmat_sym.append(temp)
            
            #TODO: Me parece que Type ya no es necesario
            data_symm.append({
                "Exc": Exc,
                "Pmat_fit": Pmat_sym,
                "Nuc": Nuc_symm,
                "Type": [new_ns, new_np, new_nd],
                "Atomic_number": data[ii]["atomic_number"],
            })

        return data_symm
            
    def _separate(self,data):
        #* Aclaracion: si usamos la base DZVP para el fitting
        #* H = (4s, 0p, 0d)
        #* C = (7s, 3p, 3d)
        #* N = (7s, 3p, 3d)
        #* O = (7s, 3p, 3d) ! checkear esta xq la diazi no tiene O
        data_sep = []

        for ii in range(len(data)):
            how_many = [0,0,0,0] # H, C, N, O
            at_no = data[ii]["Atomic_number"]
            Nucd = data[ii]["Nuc"]
            Pmat = data[ii]["Pmat_fit"]
            Exc  = data[ii]["Exc"]
            fH, fC, fN, fO = [], [], [], []

            # Contamos cuantos atomos de cada tipo hay
            for jj in range(len(at_no)):
                if at_no[jj] == 1:
                    how_many[0] += 1
                elif at_no[jj] == 6:
                    how_many[1] += 1
                elif at_no[jj] == 7:
                    how_many[2] += 1
                elif at_no[jj] == 8:
                    how_many[3] += 1

            # Separamos la Pfit en cada atom type
            for jj in range(len(Pmat)):
                atom_type = at_no[Nucd[jj]-1]
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
            #fH = torch.tensor(fH).view(how_many[0],-1)
            #fC = torch.tensor(fC).view(how_many[1],-1)
            #fN = torch.tensor(fN).view(how_many[2],-1)
            #fO = torch.tensor(fO).view(how_many[3],-1)

            data_sep.append({
                "Exc": Exc,
                "Hidrogen": fH,
                "Carbon": fC,
                "Nitrogen": fN,
                "Oxigeno": fO,
                "how_many": how_many,
            })
        
        return data_sep

    def _save_dataset(self,data):
        init = time.time()
        try:
            with open("dataset_PFit.pickle","wb") as f:
                pickle.dump(data,
                            f,pickle.HIGHEST_PROTOCOL)
        except:
            print("No se pudor escribir el",end=" ")
            print("Dataset AEV")
            exit(-1)
        print("Escritura",str(np.round(time.time()-init,2))+" s.")