
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
        sample["targets"] = torch.tensor(mol["Exc"])

        # Hidrogeno
        if how_many[0] != 0:
            f = mol["Hidrogen"]
            f = torch.tensor(f).view(how_many[0],-1)
            sample["Hidrogen"] = f

        # Carbono
        if how_many[1] != 0:
            f = mol["Carbon"]
            f = torch.tensor(f).view(how_many[1],-1)
            sample["Carbon"] = f

        # Nitrogeno
        if how_many[2] != 0:
            f = mol["Nitrogen"]
            f = torch.tensor(f).view(how_many[2],-1)
            sample["Nitrogen"] = f

        # Oxigeno
        if how_many[3] != 0:
            f = mol["Oxygen"]
            f = torch.tensor(f).view(how_many[3],-1)
            sample["Oxygen"] = f

        # TODO: por el momento creo q no es importante sacar el how_many
        return sample

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
            with open("index_test.txt", 'w') as file:
                for ii in range(nin,len(rand)):
                    file.write("%i\n" % rand[ii])
        except:
            print("No se pudo guardar los",end=" ")
            print("indices del test")
            exit(-1)
        
        # Guardamos los indices de Train y Val
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
        prop = {
            "targets": [],
            "Hidrogen": [],
            "Carbon": [],
            "Nitrogen": [],
            "Oxigen": [],
        }
        for mol in data:
            for key in mol:
                if key == "targets": 
                    prop[key].append(mol[key].item())
                else:
                    for ii in range(mol[key].shape[0]):
                        ff = mol[key][ii].tolist()
                        for jj in range(len(ff)):
                            prop[key].append(ff[jj])
        
        if len(prop["targets"]) != 0:
            prop["targets"] = torch.tensor(prop["targets"])
        if len(prop["Hidrogen"]) != 0:
            prop["Hidrogen"] = torch.tensor(prop["Hidrogen"]).view(-1,4)
        if len(prop["Carbon"]) != 0:
            prop["Carbon"] = torch.tensor(prop["Carbon"]).view(-1,13)
        if len(prop["Nitrogen"]) != 0:
            prop["Nitrogen"] = torch.tensor(prop["Nitrogen"]).view(-1,13)
        if len(prop["Oxigen"]) != 0:
            prop["Oxigen"] = torch.tensor(prop["Oxigen"]).view(-1,13)
        else:
            prop["Oxigen"] = torch.tensor([0.0])

        ff = {
            "targets": {
                "mean": prop["targets"].mean(dim=0),
                "std" : prop["targets"].std(dim=0),
            },
            "Hidrogen": {
                "mean": prop["Hidrogen"].mean(dim=0),
                "std" : prop["Hidrogen"].std(dim=0),
            },
            "Carbon": {
                "mean": prop["Carbon"].mean(dim=0),
                "std" : prop["Carbon"].std(dim=0),
            },
            "Nitrogen": {
                "mean": prop["Nitrogen"].mean(dim=0),
                "std" : prop["Nitrogen"].std(dim=0),
            },
            "Oxigen": {
                "mean": prop["Oxigen"].mean(dim=0),
                "std" : prop["Oxigen"].std(dim=0),
            },
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

            print(fact.keys())
            print(mol.keys())
            exit(-1)

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

        for mol in data:
            print(mol.keys())
            print(fact.keys())
            exit(-1)

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






