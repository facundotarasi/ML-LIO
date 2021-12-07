import time
import numpy as np
import pickle
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
        self.name_out = inputs["name_out"]
        self.save_pos = inputs["save_pos"]
        self.ndata = inputs["ndata"]
        self.path_results = inputs["path_results"]
        self.comps = inputs["comps"]
        self.coeff = inputs["coeff"]
        self.proj = inputs["proj"]
        self.AEV = inputs["AEV"]
        self.DSP = inputs["DSP"]

        # AEV variables
        if self.AEV:
            self.save_pos = True
            try:
                self.cutoff_rad = inputs["cutoff_rad"]
                self.cutoff_ang = inputs["cutoff_ang"]
                self.nradial    = inputs["nradial"]
                self.nangular   = inputs["nangular"]
                self.radial_Rs  = inputs["radial_Rs"]
                self.radial_etha= inputs["radial_etha"]
                self.angular_Rs = inputs["angular_Rs"]
                self.angular_etha=inputs["angular_etha"]
                self.angular_tetha=inputs["angular_tetha"]
            except: 
                print("Falta incluir algunas de las variables", end = " ")
                print("para generar los AEV")
                exit(-1)
        
        # DSP variables
        if self.DSP:
            self.save_pos = True
            try:
                self.rmin   = inputs["rmin"]
                self.rmax   = inputs["rmax"]
                self.nrad   = inputs["nrad"]
                self.nphi   = inputs["nphi"]
                self.ntheta = inputs["ntheta"]
            except:
                print("Falta incluir algunas de las variables", end = " ")
                print("para generar los DSP")
                exit(-1)


    def setup(self):
        # Leemos Coordenadas, Pfit, KinE, Nuc, type, how_many
        data = self._read_files()

        # Con esto simetrizamos las funciones p y d
        # de la densidad de fitting, por lo tanto la dimension se achica.
        # Esto no debe usarse si se trabaja con los DSP
        if self.coeff: data = self._symmetrize(data)

        # Separamos en tipo de atomos -> H, C, N, O
        # De esta data solo vamos a usar fdens, ya esta sorteada
        # en el orden correcto
        data = self._separate(data)

        # Generamos los AEV de posiciones
        if self.AEV: data_aev = self._get_AEVs(data)

        # Aqui generamos la union del AEV con las densidades
        if self.AEV: data = self._join_data(data,data_aev)

        # Si los AEV están desactivados, formateamos el Dataset
        # Para que conserve la misma forma
        if not self.AEV: data = self._format_data(data)

        # Generamos los DSP
        if self.DSP: data = self._density_spheres(data)

        # Guardamos el dataset
        self._save_dataset(data)

    def _read_files(self):
        """
        Este archivo tiene que leer todo, KinE, Pfit, coord.
        en que nucleo estan centradas las bases y de q tipo son (s,p,d)
        """
        data = []
        print("Leyendo los archivos...",end=" ")
        init = time.time()

        for key in self.comps:
            for ii in range(self.ndata):
            
                # Leo las coordenadas y el numero atomico
                file_name = self.path_C + key + "_" + str(ii+1) + ".xyz"
                atomic_number, positions, how_many = self._read_xyz(file_name)

                # Leo los descriptores elegidos
                file_name = self.path_P + key + "_" + str(ii+1) + ".dat"
                KinE, Pmat_fit, gtype, Nuc, proj = self._read_rhoE(file_name)

                # Guardo los datos en un solo diccionario
                single_data = {
                    "Targets": KinE,
                    "Atomic_number": atomic_number,
                    "How_many": how_many,
                    "Pmat_fit": Pmat_fit,
                    "gtype": gtype, 
                    "Nuc": Nuc,
                }

                if len(positions) != 0:
                    single_data["Positions"] = positions

                if self.proj:
                    single_data["proj"] = proj
                
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
        ? E1, KinE, Exc, En, Etot, M, Md
        ? nshelld: s, p, d, -, -,
        ? Nucd
        ? Pmat, con n_M elementos
        ? coeff, con n_Md elementos (en lio son los af)
        ? proj, con n_Md elementos
        """
        try:
            KinE, Pmat_fit, proj = [], [], []
            gtype, Nuc = [], []
            Md = -1
            with open(name) as f:
                nlines = 0
                for line in f:
                    nlines += 1
                    field = line.split()

                    # Leo la 1ra linea: saco KinE
                    if len(field) == 7:
                        KinE.append(float(field[1]))
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
                        # Leo la 6ta línea: proj
                        elif nlines == 6:
                            for ii in range(Md):
                                proj.append(float(field[ii]))
        
        except:
            print("El archivo " + name + " no existe")
            exit(-1) 

        return KinE, Pmat_fit, gtype, Nuc, proj

    def _symmetrize(self,data):
        # La densidad de fitting esta en orden
        # primero estan todas las s ( type[0] )
        # luego estan todas las p ( type[1] )
        # luego estan todas las d ( type[2] )

        print("Simetrizando los coeficientes de la base...")
        data_symm = []
        for ii in range(len(data)):
            ns   = data[ii]["gtype"][0]
            np   = data[ii]["gtype"][1]
            nd   = data[ii]["gtype"][2]

            KinE  = data[ii]["Targets"]
            Pmat = data[ii]["Pmat_fit"]
            if self.proj: 
                proj = data[ii]["proj"]
                proj_sym = []
            Pmat_sym, Nuc_sym = [], []

            # Ponemos las s
            for jj in range(0,ns):
                Nuc_sym.append(data[ii]["Nuc"][jj])
                Pmat_sym.append(Pmat[jj])
                if self.proj: proj_sym.append(proj[jj])

            # Simetrizamos las p
            for jj in range(ns,ns+np,3):
                temp  = Pmat[jj+0]**2
                temp += Pmat[jj+1]**2
                temp += Pmat[jj+2]**2
                Nuc_sym.append(data[ii]["Nuc"][jj])
                Pmat_sym.append(temp)
                if self.proj:
                    temp  = proj[jj+0]**2
                    temp += proj[jj+1]**2
                    temp += proj[jj+2]**2
                    proj_sym.append(temp)
            
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
                if self.proj:
                    temp  = proj[jj+0]**2
                    temp += proj[jj+1]**2
                    temp += proj[jj+2]**2
                    temp += proj[jj+3]**2
                    temp += proj[jj+4]**2
                    temp += proj[jj+5]**2
                    proj_sym.append(temp)

            
            data_symm.append({
                "Targets": KinE,
                "Pmat_fit": Pmat_sym,
                "Nuc": Nuc_sym,
                "Atomic_number": data[ii]["Atomic_number"],
                "Positions": data[ii]["Positions"],
                "How_many": data[ii]["How_many"],
            })

            if self.proj: data_symm[ii]["Proj"] = proj_sym

        return data_symm
            
    def _separate(self,data):
        #* Aclaracion: si usamos la base DZVP para el fitting
        #* H = (4s, 0p, 0d)
        #* C = (7s, 3p, 3d)
        #* N = (7s, 3p, 3d)
        #* O = (7s, 3p, 3d)
        print("Separando los datos...", end = " ")
        init = time.time()
        data_sep = []

        for mol in data:
            at_no = mol["Atomic_number"]
            Nucd  = mol["Nuc"]
            Pmat  = mol["Pmat_fit"]
            if self.proj: Proj  = mol["Proj"]
            KinE  = mol["Targets"]
            Pos   = mol["Positions"]
            #TODO: Si no me equivoco, estas listas nunca se usan
            fH, fC, fN, fO = [], [], [], []

            # genero el array que tiene las gaussianas separados x atomos
            nat = len(at_no)
            ff, ffsor = [], []
            if self.proj: ffp, ffpsor = [], []
            for ii in range(nat):
                ff.append([])
                ffsor.append([])
                if self.proj:
                    ffp.append([])
                    ffpsor.append([])

            for jj in range(len(Pmat)):
                at_idx = Nucd[jj] - 1
                ff[at_idx].append(Pmat[jj])
                if self.proj: ffp[at_idx].append(Proj[jj])
            
            # ff tiene dimensiones [nat][ngauss sime]
            # Ahora realizamos el sorting
            np_atno = np.array(at_no)
            sort_idx = np.argsort(np_atno)
            for ii in range(nat):
                ii_s = sort_idx[ii]
                for jj in range(len(ff[ii_s])):
                    ffsor[ii].append(ff[ii_s][jj])
                    if self.proj: ffpsor[ii].append(ffp[ii_s][jj])

            if self.proj:    
                data_sep.append({
                    "T": KinE,
                    "fdens": ffsor,
                    "Pos": Pos,
                    "atn": at_no,
                    "How_many": mol["How_many"],
                    "proj": ffpsor,
                })
            else:
                data_sep.append({
                    "T": KinE,
                    "fdens": ffsor,
                    "Pos": Pos,
                    "atn": at_no,
                    "How_many": mol["How_many"],
                })   
        
        end = time.time()
        print(str(np.round(end-init,2)) + " s.")
        return data_sep

    def _save_dataset(self,data):
        init = time.time()
        path = self.path_results + self.name_out
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
        print("Calculando los AEV...", end = " ")
        init = time.time()
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

        end = time.time()
        print(str(np.round(end-init,2)) + " s.")

        return data_aev

    def _sorting(self,data):
        data_aev = []
        for mol in data:
            # Listas
            atno = mol["atn"]
            pos  = mol["Pos"]
            Hw   = mol["How_many"]
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
            ele["Pos"] = mol1["Pos"]
            ele["Atno"] = mol2["Atno"].tolist()
            ele["How_many"] = mol2["How_many"]
            nat = len(mol2["Atno"])
            # Generamos el fg pa cada atomo en la molecula
            for jj in range(nat):
                ff = np.array(mol1["fdens"][jj])
                if self.proj:
                    ffp = np.array(mol1["proj"][jj])
                    fg_at = np.concatenate((mol2["AEV"][jj], ff, ffp))
                else:
                    fg_at = np.concatenate((mol2["AEV"][jj],ff))
                fg_mol.append(fg_at.tolist())
            ele["Fg"] = fg_mol
            data_new.append(ele)

        return data_new

    def _format_data(self, data):
        # Esta funcion deja el formato del Dataset igual que cuando se usan
        # AEV, aunque los mismos estén desactivados.

        data_sort = []
        for mol in data:
            # Listas
            fg_mol = []
            Target = mol["T"]
            atno = mol["atn"]
            Hw   = mol["How_many"]
            Pos = mol["Pos"]
            fg_coeff = mol["fdens"]
            if self.proj: fg_proj = mol["proj"]
            nat  = len(atno)

            # Primero se ordenan los números atómicos en el orden correcto

            # Numpy arrays
            np_atno = np.array(atno)
            atno_s  = np.zeros((nat),dtype=np.int32)
            np_pos  = np.array(Pos)
            pos_s = np.zeros((nat, 3))

            # Sorting
            sort_idx = np.argsort(np_atno)
            for ii in range(nat):
                idx = sort_idx[ii]
                atno_s[ii] = np_atno[idx]
                pos_s[ii,:] = np_pos[idx,:]

            # Después se concatenan los descriptores si fuera necesario

            if self.proj:
                for jj in range(nat):
                    fg_at = fg_coeff[jj] + fg_proj[jj]
                    fg_mol.append(fg_at)
            else:
                fg_mol = fg_coeff

            # Y se genera un diccionario del mismo tipo que al usar AEV

            data_sort.append({
                "T": Target,
                "Atno": atno_s.tolist(),
                "How_many": Hw,
                "Pos": pos_s.tolist(), 
                "Fg": fg_mol,
            })
            
        return data_sort 

    def _density_spheres(self, data):
        
        """
        Este método se usa para generar descriptores basados en la densidad
        electrónica en puntos seleccionados de la molécula, tomando como
        referencia su centro de masa.
        """

        # Primero leemos los coeficientes y exponentes asociados a las
        # funciones base de la base auxiliar.
        basis_data = self._get_coeff()

        # Calculamos el centro de masas de cada molécula
        data = self._get_mass_center(data)

        # Generamos para cada átomo un array con los puntos sobre los que
        # se quiere calcular la densidad electrónica
        data = self._dens_points(data)

        # Se computa la densidad electrónica en cada uno de los puntos
        # seleccionados
        data = self._eval_dens(data, basis_data)

        return data 

    def _get_coeff(self):
        init = time.time()
        print("Leyendo los coeficientes de la base...", end = " ")
        with open("DZVP", "r") as file:
            basis_data = {}
            for key in ["H", "C", "N", "O"]:
                basis_data[key] = {}
                basis_data[key]["exp"] = []
                basis_data[key]["coeff"] = []
            
            # Leemos el archivo que contiene los coeficientes y exponentes
            # asociados a las funciones de la base auxiliar para cada átomo
            
            # H
            for ii in range(5):
                line = file.readline()
                field = line.split()
                try:
                    basis_data["H"]["exp"].append(float(field[0]))
                    basis_data["H"]["coeff"].append(float(field[1]))
                except:
                    continue
            
            # C, N, O
            for key in ["C", "N", "O"]:
                for ii in range(35):
                    line = file.readline()
                    field = line.split()
                    try:
                        basis_data[key]["exp"].append(float(field[0]))
                        basis_data[key]["coeff"].append(float(field[1]))
                    except:
                        continue
        end = time.time()
        print(str( np.round(end - init,2)) + " s.")
        return basis_data 

    def _get_mass_center(self, data):
        init = time.time()
        print("Calculando los centros de masas...", end = " ")
        for mol in data:

            # Para cada molécula del DataSet calculamos el centro de masas
            atnum = mol["Atno"]
            pos = mol["Pos"]
            NN = len(atnum)
            acum = 0.
            MM = 0.
            for ii in range(NN):
                # Acumulamos el producto de la masa atómica por el vector posición
                # Y sumamos las masas atómicas para obtener la masa molecular
                pos_arr = np.array(pos[ii])
                if int(atnum[ii]) == 1:
                    acum = acum + 1.008 * pos_arr
                    MM = MM + 1.008
                if int(atnum[ii]) == 6:
                    acum = acum + 12.01 * pos_arr 
                    MM = MM + 12.01
                if int(atnum[ii]) == 7:
                    acum = acum + 14.01 * pos_arr 
                    MM = MM + 14.01
                if int(atnum[ii]) == 8:
                    acum = acum + 16.00 * pos_arr 
                    MM = MM + 16.00
            
            com = acum / MM # Centro de masas
            mol["COM"] = com.tolist()
        end = time.time()
        print(str(np.round(end - init, 2)) + " s.")
        return data 

    def _dens_points(self, data):
        init = time.time()
        print("Calculando los puntos seleccionados...", end = " ")
        for mol in data:
            mol["vector"] = []

            # Para cada átomo, calculamos el vector unitario con origen 
            # en el núcleo y extremo en el centro de masas
            NN = len(mol["Atno"])
            for ii in range(NN):
                vec = np.array(mol["COM"]) - np.array(mol["Pos"][ii])
                vec2 = vec**2
                norm = np.sqrt(vec2.sum())
                if norm == 0.: print("Cuidado: Se halló un vector al COM nulo")
                vec = vec/norm 
                mol["vector"].append(vec)
            
            # Ahora generamos listas con los parámetros necesarios para
            # obtener los puntos seleccionados
            # Esta lista tiene radios equiespaciados que se miden a partir
            # de los núcleos
            rmin = self.rmin
            rmax = self.rmax 
            nrad = self.nrad
            if rmin == 0:
                radius = [rmin + (rmax - rmin)*(i+1)/(nrad - 1) for i in range (nrad - 1)]
            else:
                radius = [rmin + (rmax - rmin)*i/(nrad - 1) for i in range(nrad - 1)]
                radius.append(rmax)

            # Lo mismo para los ángulos que se quieran barrer
            nphi = self.nphi 
            ntheta = self.ntheta
            phi = [2 * np.pi*i/nphi for i in range(nphi)]
            theta = [np.pi*i/(ntheta - 1) for i in range(ntheta - 1)]
            theta.append(np.pi)

            mol["D_Points"] = []
            # Ahora se generan los puntos (x, y, z) para cada átomo en cada
            # molécula del DataSet 
            for ii in range(NN):
                mol["D_Points"].append([])
                center = np.array(mol["Pos"][ii])
                vector = mol["vector"][ii]
                if rmin == 0:
                    mol["D_Points"][ii].append(center)
                for rr in radius:
                    for pp in phi:
                        for tt in theta:
                            point = np.zeros((3))
                            # Calculo phi y theta del vector unitario que apunta al centro de
                            # masas
                            thetacom = np.arccos(vector[2])
                            if vector[0] > 0.:
                                phicom = np.arctan(vector[1] / vector[0])
                            elif vector[0] < 0:
                                phicom = np.arctan(vector[1] / vector[0]) + np.pi
                            elif vector[0] == 0 and vector[1] > 0:
                                phicom = np.pi / 2
                            elif vector[0] == 0 and vector[1] < 0:
                                phicom = 3 * np.pi /2
                            elif vector[0] == 0 and vector[1] == 0:
                                phicom = 0.
                            
                            point[0] = center[0] + rr * np.cos(phicom + pp) * np.sin(thetacom + tt)
                            point[1] = center[1] + rr * np.sin(phicom + pp) * np.sin(thetacom + tt)
                            point[2] = center[2] + rr * np.cos(thetacom + tt)
                            mol["D_Points"][ii].append(point)
            del mol["vector"]
            del mol["COM"]
                            
        end = time.time()
        print(str(np.round(end - init, 2)) + " s.")
        return data 

    def _eval_dens(self, data, basis_data):
        init = time.time()
        print("Evaluando la densidad electrónica...", end = " ")
        """
        Este método computa la densidad electrónica de una molécula a partir de
        los coeficientes de las funciones de la base auxiliar
        """
        rmin = self.rmin 
        nrad = self.nrad 
        nphi = self.nphi 
        ntheta = self.ntheta 
        if rmin == 0.:
            ndpoints = (nrad - 1) * nphi * ntheta + 1
        else:
            ndpoints = nrad * nphi * ntheta 
        
        data_fin = []
        for mol in data:
            NN = len(mol["Atno"])
            mol["Dens"] = []
            for ii in range(NN):
                mol["Dens"].append([])
                for jj in range(ndpoints):
                    point = mol["D_Points"][ii][jj]
                    dens = 0.

                    # Se suman las contribuciones a la densidad de cada átomo
                    for kk in range(NN):
                        posit = np.array(mol["Pos"][kk])
                        if mol["Atno"][kk] == 1:
                            for ll in range(4):
                                dens = dens + mol["Fg"][kk][ll] * basis_function(int(ll), mol["Atno"][kk], point, posit, basis_data)
                        else:
                            for ll in range(34):
                                dens = dens + mol["Fg"][kk][ll] * basis_function(int(ll), mol["Atno"][kk], point, posit, basis_data)
                    
                    # Se agrega la densidad electrónica calculada a la lista para el átomo
                    # en cuestión
                    mol["Dens"][ii].append(dens)

            data_fin.append({
                "T": mol["T"],
                "Atno": mol["Atno"],
                "How_many": mol["How_many"],
                "Pos": mol["Pos"],
                "Fg": mol["Dens"],
            })
        end = time.time()
        print(str(np.round(end - init, 2)) + " s.")
        return data_fin 

def basis_function(type, elem, XYZ, X0Y0Z0, basis_data):
    
    if elem == 1:
        key = "H"
    elif elem == 6:
        key = "C"
    elif elem == 7:
        key = "N"
    elif elem == 8:
        key = "O"

    vdist2 = (XYZ - X0Y0Z0)**2
    dist2 = vdist2.sum()
    
    if 0 <= type <= 6:
        # Función base tipo s
        func = basis_data[key]["coeff"][type] * np.exp(- basis_data[key]["exp"][type] * dist2)
        return func
    elif type == 7 or type == 10 or type == 13:
        # Función base tipo px
        diff = XYZ - X0Y0Z0
        func = diff[0] * np.exp(- basis_data[key]["exp"][type] * dist2)
        func = basis_data[key]["coeff"][type] * func 
        return func 
    elif type == 8 or type == 11 or type == 14:
        # Función base tipo py
        diff = XYZ - X0Y0Z0
        func = diff[1] * np.exp(- basis_data[key]["exp"][type] * dist2)
        func = basis_data[key]["coeff"][type] * func 
        return func 
    elif type == 9 or type == 12 or type == 15:
        # Función base tipo pz
        diff = XYZ - X0Y0Z0
        func = diff[2] * np.exp(- basis_data[key]["exp"][type] * dist2)
        func = basis_data[key]["coeff"][type] * func 
        return func 
    elif type == 16 or type == 22 or type == 28:
        # Función base tipo dxx
        diff = XYZ - X0Y0Z0
        func = diff[0] * diff[0] * np.exp(- basis_data[key]["exp"][type] * dist2)
        func = basis_data[key]["coeff"][type] * func 
        return func 
    elif type == 17 or type == 23 or type == 29:
        # Función base tipo dyx
        diff = XYZ - X0Y0Z0
        func = diff[1] * diff[0] * np.exp(- basis_data[key]["exp"][type] * dist2)
        func = basis_data[key]["coeff"][type] * func 
        return func 
    elif type == 18 or type == 24 or type == 30:
        # Función base tipo yy
        diff = XYZ - X0Y0Z0
        func = diff[1] * diff[1] * np.exp(- basis_data[key]["exp"][type] * dist2)
        func = basis_data[key]["coeff"][type] * func 
        return func 
    elif type == 19 or type == 25 or type == 31:
        # Función base tipo zx
        diff = XYZ - X0Y0Z0
        func = diff[2] * diff[0] * np.exp(- basis_data[key]["exp"][type] * dist2)
        func = basis_data[key]["coeff"][type] * func 
        return func 
    elif type == 20 or type == 26 or type == 32:
        # Función base tipo zy
        diff = XYZ - X0Y0Z0
        func = diff[2] * diff[1] * np.exp(- basis_data[key]["exp"][type] * dist2)
        func = basis_data[key]["coeff"][type] * func 
        return func 
    elif type == 21 or type == 27 or type == 33:
        # Función base tipo zz
        diff = XYZ - X0Y0Z0
        func = diff[2] * diff[2] * np.exp(- basis_data[key]["exp"][type] * dist2)
        func = basis_data[key]["coeff"][type] * func 
        return func 
