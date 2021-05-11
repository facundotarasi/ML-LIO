
"""
Este module sirve para entrenar diferentes moleculas, usando 
la densidad de fitting
"""

# Predictor
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import random

# DataSet and DataLoader
import torch
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Neural Network
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Results
from scipy import stats

# H C N O
how_many = [2,1,2,0]

# Esto genera el dataset con el fingerprint
class Predictor:
    def __init__(self, inputs=None):
        self.path = inputs["path"]
        self.ndata = inputs["ndata"]
        self.atomic_number = [6,1,1,7,7]

    def setup(self):
        # Obtenemos las Exc y las Densidades de Fitt
        data = self._read_files()

        # Obtenemos en que nucleo esta centrado c/elemento
        # de la densidad de fitting
        gaussian = self._read_Nuc("Nucd_file.dat")

        # Con esto simetrizamos las funciones p y d
        # de la densidad de fitting, por lo tanto la dimension se achica
        data = self._symmetrize(data,gaussian)

        # Separamos en tipo de atomos -> H, C, N, O
        data = self._separate(data)

        self._save_dataset(data)

        # Esto grafica la distribucion de atomos
        # de las moleculas en el dataset
        #self._analysis(geom)

    def _read_files(self):
        data = []
        print("Leyendo los archivos...",end=" ")
        init = time.time()
        for ii in range(self.ndata):
            # Leo las Densidades y las Energias
            file_name = self.path + str(ii+1) + ".dat"
            single_data = self._read_one_file(file_name)

            # Guardo ambos valores
            data.append(single_data)

        fin = time.time()
        print(str(np.round(fin-init,2))+" s.")
        return data
    
    def _read_one_file(self,name):
        """
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

        res = {
            "type": gtype,
            "Nucd": Nuc,
        }
        return res

    def _symmetrize(self,data,gaussian):
        # La densidad de fittin esta en orden
        # primero estan todas las s ( type[0] )
        # luego estan todas las p ( type[1] )
        # luego estan todas las d ( type[2] )
        ns   = gaussian["type"][0]
        np   = gaussian["type"][1]
        nd   = gaussian["type"][2]

        data_symm = []
        for ii in range(len(data)):
            Exc  = data[ii]["Exc"]
            Pmat = data[ii]["Pmat_fit"]
            Pmat_sym = []
            Nuc_symm  = []
            new_ns, new_np, new_nd = 0, 0, 0

            # Ponemos las s
            for jj in range(0,ns):
                new_ns += 1
                Nuc_symm.append(gaussian["Nucd"][jj])
                Pmat_sym.append(Pmat[jj])

            # Simetrizamos las p
            for jj in range(ns,ns+np,3):
                temp  = Pmat[jj+0]**2
                temp += Pmat[jj+1]**2
                temp += Pmat[jj+2]**2
                new_np += 1
                Nuc_symm.append(gaussian["Nucd"][jj])
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
                Nuc_symm.append(gaussian["Nucd"][jj])
                Pmat_sym.append(temp)
            
            data_symm.append({
                "Exc": Exc,
                "Pmat_fit": Pmat_sym,
                "Nucd": Nuc_symm,
                "Type": [new_ns, new_np, new_nd],
            })

        return data_symm
            
    def _separate(self,data):
        #* Aclaracion: si usamos la base DZVP para el fitting
        #* H = (4s, 0p, 0d)
        #* C = (7s, 3p, 3d)
        #* N = (7s, 3p, 3d)
        #* O = (7s, 3p, 3d) ! checkear esta xq la diazi no tiene O
        data_sep = []
        at_no = self.atomic_number

        for ii in range(len(data)):
            Nucd = data[ii]["Nucd"]
            Pmat = data[ii]["Pmat_fit"]
            Exc  = data[ii]["Exc"]
            fH, fC, fN, fO = [], [], [], []

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
            fH = torch.tensor(fH).view(how_many[0],-1)
            fC = torch.tensor(fC).view(how_many[1],-1)
            fN = torch.tensor(fN).view(how_many[2],-1)
            #fO = torch.tensor(fO).view(how_many[3],-1)

            data_sep.append({
                "Exc": Exc,
                "Hidrogen": fH,
                "Carbon": fC,
                "Nitrogen": fN,
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

# Generamos el DataSet
class Dataset(torch.utils.data.Dataset):
    def __init__(self,data):
        self.my_data = data
    
    def __len__(self):
        return len(self.my_data)

    def __getitem__(self,idx):
        mol = self.my_data[idx]
        targets = torch.tensor(mol["Exc"])

        sample = {
            "targets": targets,
            "Hidrogen": mol["Hidrogen"],
            "Carbon": mol["Carbon"],
            "Nitrogen": mol["Nitrogen"],
        }

        return sample

#Definicion del DataModule
class DataModule(pl.LightningDataModule):
    # Main Methods
    def __init__(self, hooks):
        super().__init__()
        print("Init DataModule")

        self.seed = hooks["seed"]
        self.data_file = hooks["dataset_file"]        
        self.n_train_val = hooks["n_train_val"]
        self.test_size = hooks["test_size"]
        self.batch_size = hooks["batch_size"]

        # Seteamos la seed
        random.seed(self.seed)
        if self.seed != None:
            torch.manual_seed(self.seed)

    def setup(self, stage = None):
        print("Setup stage:", stage)

        # Leemos los datos en el dataset
        data = self._load_data( )

        # Checkeamos dimensiones
        self.ndata = len(data)
        if (self.ndata < self.n_train_val):
            print("La cantidad de datos para train",end=" ")
            print("es mayor a la del dataset")
            exit(-1)

        if stage == "fit":
            # Generamos una array con numeros random
            rand_number = self._get_random()

            # Separamos los datos para train_val y test
            # * Ahora data tiene solo los datos de train_val
            data = self._separate_data(data,rand_number)

            # Generamos el dataset para toda la muestra de train_val
            data_ds = Dataset(data)

            # Spliteamos la data en train y val
            train_ds, val_ds = train_test_split(data_ds, 
                       test_size=self.test_size, shuffle=True, random_state=42)


            # Obtenemos los factores de normalizacion
            self.factor_norm = self._get_norm(train_ds)

            # Normalizamos los datos de trian y validation
            self.train_ds = self._normalize(train_ds,self.factor_norm)
            self.val_ds   = self._normalize(val_ds,self.factor_norm)
        
        elif stage == "test":
            # Leo los indices empleados en test
            rand_ind = self._read_indices( )

            # Cargo los datos de test
            data = self._separate_data(data,rand_ind)

            # Generamos el dataset para toda la muestra de train_val
            test_ds = Dataset(data)

            # Leemos los factores de normalizacion
            self.factor_norm = self._read_norm( )

            # Normalizamos
            self.test_ds = self._normalize(test_ds,self.factor_norm)
            
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=6)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=6)

    def _load_data(self):
        name = self.data_file
        try:
            with open(name,"rb") as f:
                data = pickle.load(f)
        except:
            print("No se pudo leer el", end=" ")
            print("archivo", name)
            exit(-1)
        return data

    def _get_random(self):
        # Este metodo regresa una lista
        # con numeros enteros entre 0 y ndata
        # en una lista
        random.seed(self.seed)
        rand = list(range(self.ndata))
        random.shuffle(rand)
        nin = self.n_train_val

        # guardamos los indices que se van a usar
        # luego en el test
        try:
            with open("index_test.txt", 'w') as file:
                for ii in range(nin,len(rand)):
                    file.write("%i\n" % rand[ii])
        except:
            print("No se pudo guardar los",end=" ")
            print("indices del test")
            exit(-1)
        
        # Guardamos los indices de train y val para probar cosas
        try:
            with open("index_train_val.txt", 'w') as file:
                for ii in range(0,nin):
                    file.write("%i\n" % rand[ii])
        except:
            print("No se pudo guardar los",end=" ")
            print("indices del train_val")
            exit(-1)
        
        return rand[0:nin]

    def _separate_data(self,data,rand):
        ntot = len(rand)
        data_cut = []
        for ii in range(ntot):
            idx = rand[ii]
            data_cut.append(data[idx])
        
        return data_cut

    def _get_norm(self,data):
        ntot = len(data)
        mean_H = torch.zeros(4)
        mean_C = torch.zeros(13)
        mean_N = torch.zeros(13)
        mean_Exc = torch.zeros(1)

        """
        tmp = []
        for mol in data:
            var = "Nitrogen"
            for ii in range(mol[var].shape[0]):
                for jj in range(mol[var].shape[1]):
                    tmp.append(mol[var][ii,jj].item())
        
        tmp = torch.tensor(tmp).view(-1,13)
        print(tmp.mean(dim=0))
        print(tmp.std(dim=0))
        """

        # Obtenemos los promedios, en la diazi tengo 2H, 1C, 2N
        for mol in data:
            mean_H += mol["Hidrogen"][0,:]
            mean_H += mol["Hidrogen"][1,:]
            mean_C += mol["Carbon"][0,:]
            mean_N += mol["Nitrogen"][0,:]
            mean_N += mol["Nitrogen"][1,:]
            mean_Exc += mol["targets"]
        mean_H /= ntot*2
        mean_C /= ntot*1
        mean_N /= ntot*2
        mean_Exc /= ntot*1

        std_H = torch.zeros(4)
        std_C = torch.zeros(13)
        std_N = torch.zeros(13)
        std_Exc = torch.zeros(1)

        # Obtenemos la desviacion promedio
        for mol in data:
            std_H   += ( mol["Hidrogen"][0,:]-mean_H )**2
            std_H   += ( mol["Hidrogen"][1,:]-mean_H )**2
            std_C   += ( mol["Carbon"][0,:]-mean_C )**2
            std_N   += ( mol["Nitrogen"][0,:]-mean_N )**2
            std_N   += ( mol["Nitrogen"][1,:]-mean_N )**2
            std_Exc += ( mol["targets"] - mean_Exc )**2
        std_H /= (ntot*2)
        std_C /= (ntot*1)
        std_N /= (ntot*2)
        std_Exc /= (ntot*1)

        ff = {
            "mean_Exc": mean_Exc,
            "std_Exc": torch.sqrt(std_Exc),

            "mean_H": mean_H,
            "std_H": torch.sqrt(std_H),

            "mean_C": mean_C,
            "std_C": torch.sqrt(std_C),

            "mean_N": mean_N,
            "std_N": torch.sqrt(std_N),
        }

        # Escribo en un archivo
        try:
            with open("factors_norm.pickle","wb") as f:
                pickle.dump(ff, f,pickle.HIGHEST_PROTOCOL)
        except:
            print("No se pudor escribir los",end=" ")
            print("factores de normalizacion")
            exit(-1)

        return ff

    def _normalize(self,data,fact):
        data_norm = []

        for mol in data:
            # Inicializamos variables
            Exc = torch.zeros(1)
            H = torch.zeros(mol["Hidrogen"].shape[0],
                            mol["Hidrogen"].shape[1])
            C = torch.zeros(mol["Carbon"].shape[0],
                            mol["Carbon"].shape[1])
            N = torch.zeros(mol["Nitrogen"].shape[0],
                            mol["Nitrogen"].shape[1])

            Exc  = ( mol["targets"]-fact["mean_Exc"] ) 
            Exc /= fact["std_Exc"]

            H[0,:] = ( mol["Hidrogen"][0,:]-fact["mean_H"] )
            H[0,:] /= fact["std_H"]
            H[1,:] = ( mol["Hidrogen"][1,:]-fact["mean_H"] )
            H[1,:] /= fact["std_H"]

            C[0,:] = ( mol["Carbon"][0,:]-fact["mean_C"] )
            C[0,:] /= fact["std_C"]

            N[0,:] = ( mol["Nitrogen"][0,:]-fact["mean_N"] )
            N[0,:] /= fact["std_N"]
            N[1,:] = ( mol["Nitrogen"][1,:]-fact["mean_N"] )
            N[1,:] /= fact["std_N"]

            data_norm.append({
                "targets": Exc,
                "Hidrogen": H,
                "Carbon": C,
                "Nitrogen": N,
            })

        return data_norm

    def _read_indices(self):
        numbers = []
        try:
            with open("index_test.txt", 'r') as file:
                for linea in file:
                    numbers.append(int(linea))
        except:
            print("No se pudo leer el",end=" ")
            print("archivo con los indices")
            exit(-1)
        
        return numbers

    def _read_norm(self):
        try:
            with open("factors_norm.pickle","rb") as f:
                factor_norm = pickle.load(f)
        except:
            print("No se pudo leer el", end=" ")
            print("archivo de normalizacion")
            exit(-1)
        
        return factor_norm

# Modelo de cada Atom Types
class Atom_Model(pl.LightningModule):
    def __init__(self,nin,nout,Fact):
        super(Atom_Model, self).__init__()

        # Arquitectura de la NN
        self.fc1 = nn.Linear(nin,1000)
        #self.fc2 = nn.Linear(128,128)
        #self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(1000,nout)

        self.act = Fact(num_parameters=1,init=0.25)

    def forward(self,x):
        out = self.act(self.fc1(x))
        #out = self.act(self.fc2(out))
        #out = self.act(self.fc3(out))
        out = self.fc4(out)

        return out

# Definicion del Modelo
class Modelo(pl.LightningModule):
    def __init__(self,config):
        super().__init__()

        # Guardamos los hyperparametros
        self.save_hyperparameters(config)

        # Atoms Model
        self.Hydrogen  = Atom_Model(4, 1,nn.PReLU)
        self.Carbon    = Atom_Model(13,1,nn.PReLU)
        self.Nitrogen  = Atom_Model(13,1,nn.PReLU)
        #self.Oxygen    = Atom_Model(config["nin"],config["nout"],nn.LeakyReLU)

        # Loss Function
        self.err = getattr(nn,self.hparams.Loss)()

        # Logs Variables
        self.train_loss, self.val_loss = [], []
        self.pred, self.real = [], []

    def forward(self, H, C, N):
        out_H = self.Hydrogen(H)
        out_C = self.Carbon(C)
        out_N = self.Nitrogen(N)

        out_H = out_H.view(-1,2)
        out_N = out_N.view(-1,2)

        out = out_H[:,0] + out_H[:,1] + out_C[:,0]
        out += out_N[:,0] + out_N[:,1]

        # Ahora acumulo las energias para H y N
        """
        for ii in range(0,out_H.shape[0],2):
            H_accum[ii] = out_H[ii+0] + out_H[ii+1]
            N_accum[ii] = out_N[ii+0] + out_N[ii+1]
        
        out = out_C + H_accum + N_accum
        """
        out = out.view(-1,1)
        return out

    def training_step(self, batch, batch_idx):
        # Extraemos inputs y outputs
        real = batch["targets"]
        H   = batch["Hidrogen"].view(-1,4)
        C   = batch["Carbon"].view(-1,13)
        N   = batch["Nitrogen"].view(-1,13)

        pred = self(H,C,N)
        loss = self.err(pred,real)
        self.log("val_loss",loss,on_epoch=True,on_step=False,prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # Extraemos inputs y outputs
        real = batch["targets"]
        H   = batch["Hidrogen"].view(-1,4)
        C   = batch["Carbon"].view(-1,13)
        N   = batch["Nitrogen"].view(-1,13)

        pred = self(H,C,N)
        loss = self.err(pred,real)
        self.log("val_loss",loss,on_epoch=True,on_step=False,prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        # Extraemos inputs y outputs
        real = batch["targets"]
        H   = batch["Hidrogen"].view(-1,4)
        C   = batch["Carbon"].view(-1,13)
        N   = batch["Nitrogen"].view(-1,13)

        pred = self(H,C,N)

        for ii in range(pred.shape[0]):
            self.real.append(real[ii].item())
            self.pred.append(pred[ii].item())

    def training_epoch_end(self,outputs):
        val = torch.stack([x["loss"] for x in outputs]).mean()
        self.train_loss.append(val.item())

    def validation_epoch_end(self,outputs):
        val = torch.stack([x for x in outputs]).mean()
        self.val_loss.append(val.item())

    def configure_optimizers(self):
        optim = getattr(torch.optim,self.hparams.Optimizer)(self.parameters(),
                                   lr=self.hparams.lr,weight_decay=self.hparams.lr_decay)
        lr_scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer=optim, mode="min",
                         patience=3, factor=0.5, verbose=True),
            "reduce_on_plateau": True,
            "monitor": "val_loss",
        }

        return [optim], [lr_scheduler]

    def graficar(self):
        plt.title("LOSS")
        plt.plot(self.train_loss,label="train")
        plt.plot(self.val_loss,label="val")
        plt.yscale("log")
        plt.ylabel("MSE")
        plt.xlabel("Epoch")
        plt.legend()
        path = self.hparams.path_results
        plt.savefig(path + "/loss_train.png")

    def graficar_test(self,fact):
        path = self.hparams.path_results
        pred = torch.tensor(self.pred)
        real = torch.tensor(self.real)
        mean = fact["mean_Exc"]
        std2 = fact["std_Exc"]

        # Desnormalizamos
        pred = (pred * std2 + mean) * 627.5 # Kcal/mol
        real = (real * std2 + mean) * 627.5 # Kcal/mol
        pred = pred.detach().numpy()
        real = real.detach().numpy()

        # Calculamos la mae
        mae = np.abs(pred-real).mean()

        # Ajuste Lineal de los datos
        slope, intercept, r_value, p_value, std_err = \
                       stats.linregress(real,pred)

        # Graficamos los Datos
        title = "Energia Kcal/mol"
        etiqueta = "MAE " + str(np.round(mae,2)) + "Kcal/mol"
        plt.title(title)
        plt.plot(real,pred,"o",label=etiqueta)
        plt.legend()

        # Graficamos la Recta
        r_value = np.round(r_value**2,2)
        etiqueta = "R^2: " + str(r_value)
        plt.plot(real, intercept + slope*real, 'r', label=etiqueta)
        plt.legend()

        plt.xlabel("Real [Kcal/mol]")
        plt.ylabel("Pred [Kcal/mol]")
        
        # Guardamos la figura
        plt.savefig(path+"/pred_vs_real.png")

        # Guardamos los resultados
        np.savetxt(path+"/pred.txt",pred)
        np.savetxt(path+"/real.txt",real)






