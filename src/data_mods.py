import time
import numpy as np
import pickle
import numpy as np
import yaml
from matplotlib import pyplot as plt

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

        # AEV variables
        self.cutoff_rad = inputs["cutoff_rad"]
        self.cutoff_ang = inputs["cutoff_ang"]
        self.nradial    = inputs["nradial"]
        self.nangular   = inputs["nangular"]
        self.radial_Rs  = inputs["radial_Rs"]
        self.radial_etha= inputs["radial_etha"]
        self.angular_Rs = inputs["angular_Rs"]
        self.angular_etha=inputs["angular_etha"]
        self.angular_tetha=inputs["angular_tetha"]

    def setup(self):
        # Leemos Coordenadas, Pfit, Exc, Nuc, type, how_many
        data = self._read_files()

        # Con esto simetrizamos las funciones p y d
        # de la densidad de fitting, por lo tanto la dimension se achica
        data = self._symmetrize(data)

        # Separamos en tipo de atomos -> H, C, N, O
        # De esta data solo vamos a usar fdens, ya esta sorteada
        # en el orden correcto
        data = self._separate(data)

        # Generamos los AEV de posiciones
        data_aev = self._get_AEVs(data)

        # Aqui generamos la union del AEV con las densidades
        data = self._join_data(data,data_aev)

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
                "Positions": data[ii]["positions"],
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
            Pos   = mol["Positions"]
            fH, fC, fN, fO = [], [], [], []

            # genero el array que tiene las gaussianas separados x atomos
            nat = len(at_no)
            ff, ffsor = [], []
            for ii in range(nat):
                ff.append([])
                ffsor.append([])

            for jj in range(len(Pmat)):
                at_idx = Nucd[jj] - 1
                ff[at_idx].append(Pmat[jj])
            
            # ff tiene dimensiones [nat][ngauss sime]
            # Ahora realizamos el sorting
            np_atno = np.array(at_no)
            sort_idx = np.argsort(np_atno)
            for ii in range(nat):
                ii_s = sort_idx[ii]
                for jj in range(len(ff[ii_s])):
                    ffsor[ii].append(ff[ii_s][jj])

            data_sep.append({
                "T": Exc,
                "fdens": ffsor,
                "Pos": Pos,
                "atn": at_no,
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
            print("No se pudo escribir el",end=" ")
            print("Dataset AEV")
            exit(-1)
        print("Escritura",str(np.round(time.time()-init,2))+" s.")

    def _get_AEVs(self,data):
        # Primero ponemos las posiciones en el orden
        # H, C, N, O
        # data_aev: contiene todo lo necesario para
        #           generar AEVs de posiciones
        data_aev = self._sorting(data)

        # Obtengo las distancias
        data_aev = self._distances(data_aev)

        # Obtengo los cutoffs
        data_aev = self._get_cutoffs(data_aev)

        # Generamos los fingerprints de cada atomo
        # en cada molecula
        data_aev = self._fingerprint(data_aev)

        return data_aev

    def _sorting(self,data):
        data_aev = []
        for mol in data:
            # Listas
            atno = mol["atn"]
            pos  = mol["Pos"]
            Hw   = mol["how_many"]
            nat  = len(atno)

            # Numpy arrays
            np_atno = np.array(atno)
            np_pos  = np.array(pos)
            atno_s  = np.zeros((nat),dtype=np.int32)
            pos_s   = np.zeros((nat,3))

            # Sorting
            sort_idx = np.argsort(np_atno)
            for ii in range(nat):
                idx = sort_idx[ii]
                atno_s[ii] = np_atno[idx]
                pos_s[ii,:] = np_pos[idx,:]
            
            data_aev.append({
                "Atno": atno_s,
                "Pos" : pos_s,
                "How_many": Hw,
            })

        return data_aev

    def _distances(self,data):
        for ii in range(len(data)):
            mol = data[ii]
            nat = len(mol["Atno"])
            dist = np.empty((nat,nat))
            for jj in range(nat):
                dist[jj,jj] = 0.
                for kk in range(jj+1,nat):
                    at_j = mol["Pos"][jj,:]
                    at_k = mol["Pos"][kk,:]
                    diff = at_j - at_k
                    diff = np.sum(diff**2)
                    dist[jj,kk] = np.sqrt(diff)
                    dist[kk,jj] = np.sqrt(diff)
            data[ii]["Dist"] = dist
        return data

    def _get_cutoffs(self, data):
        for ii in range(len(data)):
            mol = data[ii]["Dist"]
            nat = len(data[ii]["Atno"])
            Fc_rad = np.empty((nat,nat))
            Fc_ang = np.empty((nat,nat))
            for jj in range(nat):
                Fc_rad[jj,jj] = 0.
                Fc_ang[jj,jj] = 0.
                for kk in range(jj+1,nat):
                    dist = mol[jj,kk]
                    fc_rad, fc_ang = self._single_cutoff(dist)
                    Fc_rad[jj,kk] = fc_rad
                    Fc_rad[kk,jj] = fc_rad
                    Fc_ang[jj,kk] = fc_ang
                    Fc_ang[kk,jj] = fc_ang

            data[ii]["Fc_rad"] = Fc_rad
            data[ii]["Fc_ang"] = Fc_ang
        return data

    def _single_cutoff(self,dd):
        Rc_rad = self.cutoff_rad
        Rc_ang = self.cutoff_ang
        pi = np.pi

        # Radial Cutoff
        if dd > Rc_rad:
            fc_rad = 0.
        else:
            fc_rad = 0.5 * np.cos(pi*dd/Rc_rad) + 0.5
        
        # Angular Cutoff
        if dd > Rc_ang:
            fc_ang = 0.
        else:
            fc_ang = 0.5 * np.cos(pi*dd/Rc_ang) + 0.5
        
        return fc_rad, fc_ang

    def _fingerprint(self,data):
        #TODO: CHECKEAR LA CUENTA X FAVOR DE LOS FG
        # Checkeamos dimensiones de gaussianas
        nang = len(self.angular_Rs)*len(self.angular_tetha)
        if self.nradial != len(self.radial_Rs):
            print("La cantidad de gaussianas es diferente a",end=" ")
            print("la dimension Rs radial")
            exit(-1)
        elif self.nangular != nang:
            print("La cantidad de gaussianas es diferente a",end=" ")
            print("la dimension Rs y tetha angular")
            exit(-1)
        
        # Dimension del AEV
        #! A esto hay que agregarla la cantidad de densidades luego
        total_gauss = 4 * self.nradial + 10 * self.nangular
        t_acum = 0.

        # Barremos todas las moleculas en el dataset
        for mm in range(len(data)):
            mol = data[mm]
            nat = len(mol["Atno"])
            mol_fg = np.empty((nat,total_gauss))

            # Barremos todos los atomos en la molecula
            for at in range(nat):
                # Obtenemos las contribuciones radiales
                fg_rad = self._get_fg_radial(mol,self.radial_Rs,
                               self.radial_etha,at,nat)

                # Obtenemos las contribuciones angulares
                fg_ang = self._get_fg_angular(mol,self.angular_Rs,
                               self.angular_tetha,at,nat)

                # Guardamos ambas contribuciones en el fg
                mol_fg[at,:] = np.concatenate((fg_rad,fg_ang))

                """
                #* Esto es solo para graficar los fg
                plt.plot(mol_fg[at,:],"x-",label=str(mol["Atno"][at]))
                plt.xticks(range(0,128,8))
                plt.legend()
                plt.show()
                plt.close()
                exit(-1)
                """
            data[mm]["AEV"] = mol_fg
        return data

    def _get_fg_radial(self,mol,Rs,etha,idx_at,nat):
        # El orden de la parte radial es:
        # H, C, N, O
        ngauss = self.nradial
        fg   = np.zeros((4,ngauss))
        Zat  = mol["Atno"]
        dist = mol["Dist"]
        Fc   = mol["Fc_rad"]
        how_many = mol["How_many"]

        # Barremos todas las gaussianas
        for gau in range(ngauss):
            start = 0
            # Barremos todos los atom types
            for at_type in range(4):
                # Barremos todos los atomos dentro de ese type
                for kk in range(start,start+how_many[at_type]):
                    if idx_at == kk:
                        continue
                    # Calculamos la parte radial
                    val = (dist[idx_at,kk]-Rs[gau])**2 * etha
                    val = np.exp(-val) * Fc[idx_at,kk]
                    fg[at_type,gau] += val
                start += how_many[at_type]
        fg = fg.reshape((4*ngauss))
        return fg

    def _get_fg_angular(self,mol,Rs,tetha,idx_at,nat):
        # El orden de la parte angular es
        # HH, HC, HN, HO, CC, CN, CO, NN, NO, OO
        ngauss = self.nangular
        fg   = np.zeros((10,ngauss))
        Zat  = mol["Atno"]
        dist = mol["Dist"]
        pos  = mol["Pos"]
        Fc   = mol["Fc_ang"]
        etha_ang = self.angular_etha
        etha_rad = self.radial_etha
        how_many = mol["How_many"]

        # Obtenemos la matriz de angulas X - idx_at - Y
        Ang = self._get_angles(dist,pos,Zat,idx_at)

        # TODO: esto hay que hacerlo en el init, esta operacion
        # TODO: es la misma pa todos los atomos en todas las molec.
        # Metemos todos los parametros de la parte angular
        # en una sola lista
        gauss_param = []
        for ii in range(len(Rs)):
            for jj in range(len(tetha)):
                lista = [Rs[ii],tetha[jj]]
                gauss_param.append(lista)
        if (len(gauss_param)) != ngauss:
            print("Error en la lista de parametros y la ",end=" ")
            print("cantidad de gaussianas")
            exit(-1)
        
        # Barremos las gaussianas
        for gi in range(len(gauss_param)):
            start_i, start_j = 0, 0
            ang_type = 0
            p_Rs = gauss_param[gi][0]
            p_tetha = gauss_param[gi][1]
            # Barremos el primer atom type
            for at_i in range(4):
                # Barremos el segundo atom type
                for at_j in range(at_i,4):
                    # Barremos los atomos dentro de type i
                    for ii in range(0,how_many[at_i]):
                        ii_x = start_i + ii
                        if idx_at == ii_x:
                            continue
                        init = 0
                        if at_i == at_j:
                            init = ii+1
                        # Barremos los atomos dentro de type j
                        for jj in range(init,how_many[at_j]):
                            jj_x = start_j + jj
                            if idx_at == jj_x:
                                continue

                            # Primer Termino
                            term_1 = np.cos(Ang[ii_x,jj_x]-p_tetha)
                            term_1 = (term_1 + 1.)**etha_ang

                            # Segundo Termino
                            term_2 = dist[idx_at,ii_x] + dist[idx_at,jj_x]
                            term_2 = ((term_2/2.)-p_Rs)**2
                            term_2 = np.exp(-etha_rad*term_2)

                            # Final Termino
                            final = Fc[idx_at,ii_x]*Fc[idx_at,jj_x]
                            final = term_1 * term_2 * final
                            final = 2.**(1.-etha_ang) * final

                            # Acumulo 
                            fg[ang_type,gi] += final

                    start_j  += how_many[at_j]
                    ang_type += 1
                start_i += how_many[at_i]
                start_j  = how_many[at_i]
        
        fg = fg.reshape(10*ngauss)
        return fg

    def _get_angles(self,dist,pos,Zat,idx_at):
        # Formula: cos alfa = (a*b)/ (|a|*|a|)
        nat = len(Zat)
        mat = np.zeros((nat,nat))
        for ii in range(nat):
            if ii == idx_at:
                continue
            for jj in range(ii+1,nat):
                if jj == idx_at:
                    continue
                # vector idx_at - ii
                list_i = [pos[idx_at,0]-pos[ii,0]]
                list_i.append(pos[idx_at,1]-pos[ii,1])
                list_i.append(pos[idx_at,2]-pos[ii,2])
                vec_i = np.array(list_i)

                # vector idx_at - jj
                list_j = [pos[idx_at,0]-pos[jj,0]]
                list_j.append(pos[idx_at,1]-pos[jj,1])
                list_j.append(pos[idx_at,2]-pos[jj,2])
                vec_j = np.array(list_j)

                numerator = (vec_i*vec_j).sum()
                denominator = dist[idx_at,ii]*dist[idx_at,jj]
                val = numerator / denominator
                mat[ii,jj] = np.arccos(val)
                mat[jj,ii] = np.arccos(val)
                # esta en radianes, para pasar a grados hay x 57.2958
        return mat

    def _join_data(self,data,data_aev):
        data_new = []
        if len(data) != len(data_aev):
            print("Los 2 datasets difieren en tamaño")
            exit(-1)

        for ii in range(len(data)):
            mol1 = data[ii]
            mol2 = data_aev[ii]
            fg_mol = []
            ele = {}
            ele["T"] = mol1["T"]
            ele["Atno"] = mol2["Atno"].tolist()
            ele["How_many"] = mol2["How_many"]
            nat = len(mol2["Atno"])
            # Generamos el fg pa cada atomo en la molecula
            for jj in range(nat):
                ff = np.array(mol1["fdens"][jj])
                fg_at = np.concatenate((mol2["AEV"][jj],ff))
                fg_mol.append(fg_at.tolist())
            ele["Fg"] = fg_mol
            data_new.append(ele)

        return data_new
        












