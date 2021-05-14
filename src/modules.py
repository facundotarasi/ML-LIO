
"""
Este module sirve para entrenar diferentes moleculas, usando 
la densidad de fitting
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import random
import torch
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy import stats

# Generamos el DataSet: 
class Dataset(torch.utils.data.Dataset):
    def __init__(self,data):
        self.my_data = data
    
    def __len__(self):
        return len(self.my_data)

    def __getitem__(self,idx):
        mol = self.my_data[idx]
        how_many = mol["how_many"]
        sample = {}

        # Real Truth
        sample["T"] = torch.tensor(mol["T"])

        # Hidrogeno
        sample["H"] = self.get_list(how_many[0],mol["H"])
        
        # Carbono
        sample["C"] = self.get_list(how_many[1],mol["C"])

        # Nitrogeno
        sample["N"] = self.get_list(how_many[2],mol["N"])

        # Oxigeno
        sample["O"] = self.get_list(how_many[3],mol["O"])

        # TODO How Many: Por el momento lo guardo despues veo
        # TODO: si es necesario o no
        sample["how_many"] = torch.tensor(how_many)

        return sample
    
    def get_list(self,nn,ll):
        if nn != 0:
            f = torch.tensor(ll).view(nn,-1)
        else:
            f = torch.tensor([])
        return f

def collate_fn(batch):
    tmp = {
        "T": [],
        "H": [],
        "C": [],
        "N": [],
        "O": [],
        "how_many": [],
    }
    for mol in batch:
        tmp["T"].append(mol["T"])
        tmp["how_many"].append(mol["how_many"])

        for key in ["H","C","N","O"]:
            prop = mol[key]
            if len(prop) != 0:
                nat = prop.shape[0]
                for ii in range(nat):
                    tmp[key].append(prop[ii,:])

    # En caso de q en el batch ninguna molecula tenga algun atomo
    for key in tmp:
        if len(tmp[key]) != 0:
            tmp[key] = torch.stack(tmp[key],dim=0)
        else:
            tmp[key] = torch.tensor([])

    return tmp

# Definicion del DataModule
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
        self.path_dir = hooks["path_dir"]
        self.mode = hooks["mode"]

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

            # Separamos los datos para train_val
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
            if self.mode == "test":
                # Leo los indices empleados en test
                rand_ind = self._read_indices( )

                # Cargo los datos de test
                data = self._separate_data(data,rand_ind)
            else:
                print("Mode in: " + self.mode)

            # Generamos el dataset para toda la muestra de train_val
            test_ds = Dataset(data)

            # Leemos los factores de normalizacion
            self.factor_norm = self._read_norm( )

            # Normalizamos
            self.test_ds = self._normalize(test_ds,self.factor_norm)
            
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, collate_fn=collate_fn, num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, collate_fn=collate_fn, num_workers=6)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          shuffle=False, collate_fn=collate_fn, num_workers=6)

    def _load_data(self):
        print("Leyendo el DataSet...", end=" ")
        init = time.time()
        name = self.data_file
        try:
            with open(name,"rb") as f:
                data = pickle.load(f)
        except:
            print("No se pudo leer el", end=" ")
            print("archivo", name)
            exit(-1)
        fin = time.time()
        print(str(np.round(fin-init,2))+" s.")
        
        return data

    def _get_random(self):
        # Este metodo regresa una lista
        # con numeros enteros entre 0 y ndata
        # en una lista
        random.seed(self.seed)
        rand = list(range(self.ndata))
        random.shuffle(rand)
        nin = self.n_train_val

        # Guardamos los indices de Test
        try:
            name = self.path_dir + "index_test.txt"
            with open(name, 'w') as file:
                for ii in range(nin,len(rand)):
                    file.write("%i\n" % rand[ii])
        except:
            print("No se pudo guardar los",end=" ")
            print("indices del test")
            exit(-1)
        
        # Guardamos los indices de Train y Val
        try:
            name = self.path_dir + "index_train_val.txt"
            with open(name, 'w') as file:
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
        prop = {
            "T": [],
            "H": [],
            "C": [],
            "N": [],
            "O": [],
        }
        for mol in data:
            for key in mol:
                if key == "T": 
                    prop[key].append(mol[key].item())
                elif key != "how_many":
                    for ii in range(mol[key].shape[0]):
                        ff = mol[key][ii].tolist()
                        for jj in range(len(ff)):
                            prop[key].append(ff[jj])
        
        # Target
        if len(prop["T"]) != 0:
            prop["T"] = torch.tensor(prop["T"])
        else:
            print("No hay ningun Real Truth")
            exit(-1)

        # Hidrogeno
        if len(prop["H"]) != 0:
            prop["H"] = torch.tensor(prop["H"]).view(-1,4)
        else:
            prop["H"] = torch.tensor([0.])

        # Carbono
        if len(prop["C"]) != 0:
            prop["C"] = torch.tensor(prop["C"]).view(-1,13)
        else:
            prop["C"] = torch.tensor([0.])

        # Nitrogeno
        if len(prop["N"]) != 0:
            prop["N"] = torch.tensor(prop["N"]).view(-1,13)
        else:
            prop["N"] = torch.tensor([0.])

        # Oxigeno
        if len(prop["O"]) != 0:
            prop["O"] = torch.tensor(prop["O"]).view(-1,13)
        else:
            prop["O"] = torch.tensor([0.])

        ff = {
            "T": {
                "mean": prop["T"].mean(dim=0),
                "std" : prop["T"].std(dim=0),
            },
            "H": {
                "mean": prop["H"].mean(dim=0),
                "std" : prop["H"].std(dim=0),
            },
            "C": {
                "mean": prop["C"].mean(dim=0),
                "std" : prop["C"].std(dim=0),
            },
            "N": {
                "mean": prop["N"].mean(dim=0),
                "std" : prop["N"].std(dim=0),
            },
            "O": {
                "mean": prop["O"].mean(dim=0),
                "std" : prop["O"].std(dim=0),
            },
        }

        # Guardo los factores de Norm en un archivo
        try:
            name = self.path_dir + "factors_norm.pickle"
            with open(name,"wb") as f:
                pickle.dump(ff, f,pickle.HIGHEST_PROTOCOL)
        except:
            print("No se pudo escribir los",end=" ")
            print("factores de normalizacion")
            exit(-1)

        return ff

    def _normalize(self,data,fact):
        data_norm = []

        for mol in data:
            mol_norm = {}

            # Normalizo el Real Truth
            Tn = (mol["T"] - fact["T"]["mean"]) / fact["T"]["std"]
            mol_norm["T"] = Tn

            # Normalizamos los atomos
            for key in ["H","C","N","O"]:
                nat = len(mol[key])
                if nat != 0:
                    ff = torch.zeros(nat,mol[key].shape[1])
                    for ii in range(nat):
                        ff[ii,:] = (mol[key][ii,:]-fact[key]["mean"])
                        ff[ii,:] /= fact[key]["std"]

                else:
                    ff = torch.tensor([])
                
                mol_norm[key] = ff
            
            # Agrego How Many
            mol_norm["how_many"] = mol["how_many"]

            # Genero el arreglo normalizado
            data_norm.append(mol_norm)

            """
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
            """

        return data_norm

    def _read_indices(self):
        numbers = []
        name = self.path_dir + "index_test.txt"
        try:
            with open(name, 'r') as file:
                for linea in file:
                    numbers.append(int(linea))
        except:
            print("No se pudo leer el",end=" ")
            print("archivo " + name)
            exit(-1)
        
        return numbers

    def _read_norm(self):
        name = self.path_dir + "factors_norm.pickle"
        try:
            with open(name,"rb") as f:
                factor_norm = pickle.load(f)
        except:
            print("No se pudo leer el", end=" ")
            print("archivo " + name)
            exit(-1)
        
        return factor_norm

# Modelo de cada Atom Types
class Atom_Model(pl.LightningModule):
    def __init__(self,nin):
        super(Atom_Model, self).__init__()

        # Arquitectura de la NN
        self.fc1 = nn.Linear(nin,1000)
        #self.fc2 = nn.Linear(128,128)
        #self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(1000,1)

        self.act = nn.PReLU(num_parameters=1,init=0.25)

    def forward(self,x):
        if len(x) != 0:
            out = self.act(self.fc1(x))
            #out = self.act(self.fc2(out))
            #out = self.act(self.fc3(out))
            out = self.fc4(out)
        else:
            out = torch.tensor([])

        return out

# Definicion del Modelo
class Modelo(pl.LightningModule):
    def __init__(self,config):
        super().__init__()

        # Guardamos los hyperparametros
        self.save_hyperparameters(config)

        # Atoms Model
        self.Hydrogen  = Atom_Model(4)
        self.Carbon    = Atom_Model(13)
        self.Nitrogen  = Atom_Model(13)
        self.Oxygen    = Atom_Model(13)

        # Loss Function
        self.err = getattr(nn,self.hparams.Loss)()

        # Logs Variables
        self.train_loss, self.val_loss = [], []
        self.pred, self.real = [], []

    def forward(self, H, C, N, O, Hw):
        out_H = self.Hydrogen(H)
        out_C = self.Carbon(C)
        out_N = self.Nitrogen(N)
        out_O = self.Oxygen(O)
        nmol  = Hw.shape[0]
        out   = torch.zeros(nmol,1)

        # Sumo los resultados
        # Hidrogeno
        if len(out_H) != 0:
            start = 0
            for mol in range(nmol):
                for at in range(start,start+Hw[mol,0]):
                    out[mol] += out_H[at]
                start += Hw[mol,0].item()
        
        # Carbono
        if len(out_C) != 0:
            start = 0
            for mol in range(nmol):
                for at in range(start,start+Hw[mol,1]):
                    out[mol] += out_C[at]
                start += Hw[mol,1].item()
        
        # Nitrogeno
        if len(out_N) != 0:
            start = 0
            for mol in range(nmol):
                for at in range(start,start+Hw[mol,2]):
                    out[mol] += out_N[at]
                start += Hw[mol,2].item()

        # Oxigeno
        if len(out_O) != 0:
            start = 0
            for mol in range(nmol):
                for at in range(start,start+Hw[mol,3]):
                    out[mol] += out_O[at]
                start += Hw[mol,3].item()

        return out

    def training_step(self, batch, batch_idx):
        # Extraemos inputs y outputs
        real = batch["T"]
        H   = batch["H"]
        C   = batch["C"]
        N   = batch["N"]
        O   = batch["O"]
        Hw  = batch["how_many"]

        pred = self(H,C,N,O,Hw)
        loss = self.err(pred,real)
        self.log("train_loss",loss,on_epoch=True,on_step=False,prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # Extraemos inputs y outputs
        real = batch["T"]
        H   = batch["H"]
        C   = batch["C"]
        N   = batch["N"]
        O   = batch["O"]
        Hw  = batch["how_many"]

        pred = self(H,C,N,O,Hw)
        loss = self.err(pred,real)
        self.log("val_loss",loss,on_epoch=True,on_step=False,prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        # Extraemos inputs y outputs
        real = batch["T"]
        H   = batch["H"]
        C   = batch["C"]
        N   = batch["N"]
        O   = batch["O"]
        Hw  = batch["how_many"]

        pred = self(H,C,N,O,Hw)

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
        mean = fact["T"]["mean"]
        std2 = fact["T"]["std"]

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






