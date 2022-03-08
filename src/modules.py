
"""
Este modulo sirve para entrenar diferentes moleculas, usando 
la densidad electronica y sus gradientes
"""

from cmath import sqrt
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy import stats
import yaml
import os 

# Leemos el yaml file de input
def read_input(files):
    try:
        with open(files, 'r') as f:
            inp = yaml.load(f,Loader=yaml.FullLoader)
    except:
        print("El archivo " + files + " no existe")
        exit(-1)
    return inp

# Generamos el DataSet: 
class DatasetDens(torch.utils.data.Dataset):
    def __init__(self,data, grid_size):
        self.my_data = data
        self.grid_size = grid_size 
    
    def __len__(self):
        return len(self.my_data)

    def __getitem__(self,idx):
        mol = self.my_data[idx]
        how_many = mol["How_many"]
        atomic = mol["Atno"]
        size = self.grid_size 
        sample = {}
        fH, fC = [], []
        fN, fO = [], []

        # Real Truth
        sample["T"] = mol["T"]

        for jj in range(atomic.shape[0]):
            if int(atomic[jj].item()) == 1:
                stack = torch.stack(([mol["Dens"][jj], mol["Gradx"][jj], mol["Grady"][jj], mol["Gradz"][jj]]))
                fH.append(torch.unsqueeze(stack, 0))
            elif int(atomic[jj].item()) == 6:
                stack = torch.stack(([mol["Dens"][jj], mol["Gradx"][jj], mol["Grady"][jj], mol["Gradz"][jj]]))
                fC.append(torch.unsqueeze(stack, 0))
            elif int(atomic[jj].item()) == 7:
                stack = torch.stack(([mol["Dens"][jj], mol["Gradx"][jj], mol["Grady"][jj], mol["Gradz"][jj]]))
                fN.append(torch.unsqueeze(stack, 0))
            elif int(atomic[jj].item()) == 8:
                stack = torch.stack(([mol["Dens"][jj], mol["Gradx"][jj], mol["Grady"][jj], mol["Gradz"][jj]]))
                fO.append(torch.unsqueeze(stack, 0))
            else:
                print("Sólo se permiten los elmentos C, H, O, N")
                exit(-1)
        
        # Hidrogeno
        if len(fH) == 0:
            sample["H"] = torch.rand(0, 4, size, size, size)
        else:
            sample["H"] = torch.cat(fH)
        
        # Carbono
        if len(fC) == 0:
            sample["C"] = torch.rand(0, 4, size, size, size)
        else:
            sample["C"] = torch.cat(fC)

        # Nitrogeno
        if len(fN) == 0:
            sample["N"] = torch.rand(0, 4, size, size, size)
        else:
            sample["N"] = torch.cat(fN)

        # Oxigeno
        if len(fO) == 0:
            sample["O"] = torch.rand(0, 4, size, size, size)
        else:
            sample["O"] = torch.cat(fO) 

        # TODO How Many: Por el momento lo guardo despues veo
        # TODO: si es necesario o no
        sample["how_many"] = how_many

        return sample

class DatasetCoeff(torch.utils.data.Dataset):
    def __init__(self, data):
        self.my_data = data
    
    def __len__(self):
        return len(self.my_data)

    def __getitem__(self,idx):
        mol = self.my_data[idx]
        how_many = mol["How_many"]
        atomic = mol["Atno"]
        Nucd = mol["Nucd"]
        coeff = mol["Coeff"]
        proj = mol["Proj"]
        sample = {}
        ff = {1: [], 6: [], 7: [], 8: []}

        # Real Truth
        sample["T"] = mol["T"]

        for jj in range(len(atomic)):
            for key in ff:
                if atomic[jj] == key:
                    atomc = []
                    atomp = []
                    for kk in range(len(Nucd)):
                        if Nucd[kk] == (jj + 1):
                            atomc.append(coeff[kk].item())
                            atomp.append(proj[kk].item())
                    stack = torch.stack((torch.tensor(atomc), torch.tensor(atomp)))
                    ff[key].append(torch.unsqueeze(stack, 0))

        # Hidrogeno
        if len(ff[1]) == 0:
            sample["H"] = torch.rand(0, 2, 4)
        else:
            sample["H"] = torch.cat(ff[1])
        
        # Carbono
        if len(ff[6]) == 0:
            sample["C"] = torch.rand(0, 2, 34)
        else:
            sample["C"] = torch.cat(ff[6])

        # Nitrogeno
        if len(ff[7]) == 0:
            sample["N"] = torch.rand(0, 2, 34)
        else:
            sample["N"] = torch.cat(ff[7])

        # Oxigeno
        if len(ff[8]) == 0:
            sample["O"] = torch.rand(0, 2, 34)
        else:
            sample["O"] = torch.cat(ff[8]) 

        sample["how_many"] = how_many

        return sample

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
        tmp["T"].append(mol["T"].unsqueeze(0))
        tmp["how_many"].append(mol["how_many"].unsqueeze(0))

        for key in ["H","C","N","O"]:
            tmp[key].append(mol[key])
    
    for key in tmp:
        tmp[key] = torch.cat(tmp[key])

    return tmp

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {x: to_device(data[x], device) for x in data}
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

# Definicion del DataModule
class DataModule(pl.LightningDataModule):
    def __init__(self, hooks):
        super().__init__()
        print("Inicializando DataModule")

        self.seed = hooks["seed"]
        self.path_dir = hooks["path_dir"]
        self.batch_size = hooks["batch_size"]
        self.indexes = hooks["indexes"]
        self.grid_size = hooks["grid_size"]
        if self.grid_size != False: self.transform = hooks["transform"]
        self.coeff = hooks["coeff"]

        # Seteamos la seed
        if self.seed != None:
            torch.manual_seed(self.seed)

    def setup(self, stage = None):

        # Si estamos durante entrenamiento, cargamos los datos de los
        # DataSets de entrenamiento y validacion
        if stage == "fit":

            print("Preparando para el entrenamiento...")

            # Cargamos los datos de training y validation
            train_data = self._load_data("Training")
            val_data = self._load_data("Validacion")

            # Generamos los DataSets de training y validation
            init = time.time()
            print("Generando los DataSets...", end = " ")

            if self.grid_size != False:
                train_ds = DatasetDens(train_data, self.grid_size)
                val_ds   = DatasetDens(val_data, self.grid_size)
            elif self.coeff == True:
                train_ds = DatasetCoeff(train_data)
                val_ds   = DatasetCoeff(val_data)

            print(str(np.round(time.time() - init, 2)) + " s.")

            # Realizamos las transformaciones si es necesario
            if self.grid_size != False:
                if self.transform:
                    train_ds = self._transform(train_ds)
                    val_ds = self._transform(val_ds)

            # Obtenemos o leemos los factores de normalizacion
            if self.grid_size != False:
                if self.transform:
                    ffee = self.path_dir + "factors_norm_trans.pickle"
                else:
                    ffee = self.path_dir + "factors_norm_untr.pickle"
            else:
                ffee = self.path_dir + "factors_norm.pickle"

            if os.path.isfile(ffee):
                self.factor_norm = self._read_norm()
            else:
                if self.grid_size != False:
                    self.factor_norm = self._get_norm_dens(train_ds)
                else:
                    self.factor_norm = self._get_norm_coeff(train_ds)

            # Normalizamos los datos de train y validation
            if self.grid_size != False:
                self.train_ds = self._normalize_dens(train_ds,self.factor_norm)
                self.val_ds   = self._normalize_dens(val_ds,self.factor_norm)
            else:
                self.train_ds = self._normalize_coeff(train_ds,self.factor_norm)
                self.val_ds   = self._normalize_coeff(val_ds,self.factor_norm)
        
        elif stage == "test":

            # Cargamos los datos de test
            test_data = self._load_data("Test")

            # Generamos el dataset para toda los datos de test
            init = time.time()
            print("Generando el DataSet...", end = " ")

            if self.grid_size != False:
                test_ds = DatasetDens(test_data, self.grid_size)
            elif self.coeff == True:
                test_ds = DatasetCoeff(test_data)

            print(str(np.round(time.time() - init, 2)) + " s.")

            # Realizamos las transformaciones si es necesario
            if self.grid_size != False:
                if self.transform:
                    test_ds = self._transform(test_ds)

            # Leemos los factores de normalizacion
            self.factor_norm = self._read_norm()

            # Normalizamos
            if self.grid_size != False:
                self.test_ds = self._normalize_dens(test_ds,self.factor_norm)
            elif self.coeff == True:
                self.test_ds = self._normalize_coeff(test_ds, self.factor_norm)
            
    def train_dataloader(self):
        return DeviceDataLoader(DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, collate_fn=collate_fn, num_workers=2), 
                          device = get_default_device())

    def val_dataloader(self):
        return DeviceDataLoader(DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, collate_fn=collate_fn, num_workers=2),
                          device = get_default_device())

    def test_dataloader(self):
        return DeviceDataLoader(DataLoader(self.test_ds, batch_size=self.batch_size,
                          shuffle=False, collate_fn=collate_fn, num_workers=2),
                          device = get_default_device())

    def _load_data(self, currstat):
        init = time.time()
        print("Leyendo los archivos de " + currstat + "...", end = " ")
        # Inicializamos una lista para los datos
        list_data = []
        for jj in self.indexes:
            first_ind = int(jj.split()[0])
            last_ind = int(jj.split()[1])

            # Abrimos cada archivo, y cargamos sus datos en las listas
            path = self.path_dir 
            if currstat == "Training":
                namef = "train_data_"
            elif currstat == "Validacion":
                namef = "val_data_"
            elif currstat == "Test":
                namef = "test_data_"

            with open(path + namef + str(first_ind) + "_to_" + \
                str(last_ind) + ".pickle", "rb") as ff:
                data_temp = pickle.load(ff)
            list_data += data_temp 
            del data_temp 

        print(str(np.round(time.time() - init, 2)) + " s.")
        return list_data 

    def _transform(self, data):
        init = time.time()
        print("Transformando las variables de input...", end = " ")
        data_trans = []
        const = 2 * (3 * (np.pi)**2 )**(1/3)

        for mol in data:
            mol_trans = {}

            # Ni el Target ni how_many se transforman, por lo que se igualan
            # a los datos originales.
            mol_trans["T"] = mol["T"]
            mol_trans["how_many"] = mol["how_many"] 

            # Primero transformamos el gradiente de la densidad en el gradiente
            # reducido (s)
            # s = |\nabla \rho| / [ 2 * (3 * \pi^2)^{1/3} \rho^{4/3} ]
            for key in ["H", "C", "N", "O"]:
                rho_43 = (mol[key][:, 0, :, :, :])**(4/3)
                rho_43 = torch.stack((rho_43, rho_43, rho_43), dim = 1)
                s = torch.abs(mol[key][:, 1:, :, :, :]) / const 
                s = s / rho_43 

                # Ahora transformamos la densidad
                # rhot = rho^{1/3}
                rho_13 = (mol[key][:, 0, :, :, :])**(1/3)

                # Y juntamos todo en un tensor
                mol_trans[key] = torch.cat((rho_13.unsqueeze(1), s), 1)

                # Tomamos logaritmo decimal
                mol_trans[key] = torch.log10(mol_trans[key]) 

                # Eliminamos valores 'nan'
                mol_trans[key] = torch.nan_to_num(mol_trans[key], nan = 0., posinf = 0., neginf = 0.)

            data_trans.append(mol_trans)

        print(str(np.round(time.time() - init, 2)) + " s.")
        return data_trans
    
    def _get_norm_dens(self,data):
        init = time.time()
        print("Calculando los factores de normalizacion...", end = " ")
        size = self.grid_size 
        fact = {
            "T": {},
            "H": {},
            "C": {},
            "N": {},
            "O": {},
        }

        acum = {
            "T": [],
            "H": [],
            "C": [], 
            "N": [],
            "O": [],
        }

        for mol in data:
            for key in mol:
                if key == "T":
                    acum[key].append(torch.unsqueeze(mol[key], 0))
                elif key != "how_many":
                    acum[key].append(mol[key])

        for key in acum:
            acum[key] = torch.cat(acum[key])

        # El diccionario acum acumula la informacion de todo el dataset.
        # Ahora hay que calcular las medias y desviaciones de cada feature
        # para cada elemento.

        # Calculamos la media y la desviacion estandar del target
        fact["T"]["mean"] = acum["T"].mean()
        fact["T"]["std"]  = acum["T"].std()

        # Calculamos media y desviacion estandar para cada elemento, y cada una
        # de las features (o sea, densidad y sus gradientes)
        for key in ["H", "C", "N", "O"]:
            # Primero promediamos sobre los atomos
            overat = acum[key].mean(0)
            # Y despues sobre las grillas
            fact[key]["mean"] = torch.rand(4)
            fact[key]["std"]  = torch.rand(4)
            for ii in range(4):
                fact[key]["mean"][ii] = overat[ii].mean().item()
                fact[key]["std"][ii]  = overat[ii].std().item()

        # Guardamos los factores de normalizacion en un archivo
        try:
            if self.transform:
                name = self.path_dir + "factors_norm_trans.pickle"
            else:
                name = self.path_dir + "factors_norm_untr.pickle"
            with open(name,"wb") as f:
                pickle.dump(fact, f,pickle.HIGHEST_PROTOCOL)
        except:
            print("No se pudo escribir los",end=" ")
            print("factores de normalizacion")
            exit(-1)

        print(str(np.round(time.time() - init, 2)) + " s.")
        return fact

    def _get_norm_coeff(self, data):
        init = time.time()
        print("Calculando los factores de normalización", end = " ")
        tempfact = {
            "T": {"mean": [], "std": []},
            "H": {"mean": [], "std": []}, 
            "C": {"mean": [], "std": []},
            "N": {"mean": [], "std": []}, 
            "O": {"mean": [], "std": []},
            "count": [],
        }

        fact = {
            "T": {"mean": [], "std": []},
            "H": {"mean": [], "std": []}, 
            "C": {"mean": [], "std": []},
            "N": {"mean": [], "std": []}, 
            "O": {"mean": [], "std": []},
        }

        acum = {
            "T": [],
            "H": [], 
            "C": [],
            "N": [], 
            "O": [],
        }

        # Para los datos, acumulamos targets y features en el diccionario acum
        count = 0
        for mol in data:
            count += 1
            for key in mol:
                if key == "T":
                    acum["T"].append(torch.unsqueeze(mol["T"],0))
                elif key != "how_many":
                    acum[key].append(mol[key])
            if (count % 2500) == 0:
                for key in acum:
                    acum[key] = torch.cat(acum[key])
                for key in acum:
                    tempfact[key]["mean"].append(acum[key].mean(dim = 0))
                    tempfact[key]["std"].append(acum[key].std(dim = 0))

                tempfact["count"].append(count)
                count = 0
                del(acum)
                acum = {
                    "T": [],
                    "H": [], 
                    "C": [],
                    "N": [], 
                    "O": [],
                }
        
        if (len(acum["T"]) != 0):
            for key in acum:
                    acum[key] = torch.cat(acum[key])
            for key in acum:
                tempfact[key]["mean"].append(acum[key].mean(dim = 0))
                tempfact[key]["std"].append(acum[key].std(dim = 0))
            
            tempfact["count"].append(count)
            del(acum)
        
        for key in tempfact:
            if key != "count":
                acmean = 0
                bigsum = 0
                for ii in range(len(tempfact[key]["mean"])):
                    acmean += tempfact[key]["mean"][ii] * tempfact["count"][ii]
                    bigsum += tempfact["count"][ii]
                fact[key]["mean"] = acmean / bigsum
                
                acstd  = 0
                bigsum = 0
                for ii in range(len(tempfact[key]["std"])):
                    acstd  += ((tempfact[key]["std"][ii])**2 + (tempfact[key]["mean"][ii] - fact[key]["mean"])**2) * (tempfact["count"][ii] - 1)
                    bigsum += tempfact["count"][ii]
                fact[key]["std"] = torch.sqrt(acstd / (bigsum - 1))  

        # Guardamos los factores de normalizacion en un archivo
        try:
            name = self.path_dir + "factors_norm.pickle"
            with open(name,"wb") as f:
                pickle.dump(fact, f,pickle.HIGHEST_PROTOCOL)
        except:
            print("No se pudo escribir los",end=" ")
            print("factores de normalizacion")
            exit(-1)

        print(str(np.round(time.time() - init, 2)) + " s.")
        return fact

    def _normalize_dens(self,data,fact):
        init = time.time()
        print("Normalizando los datos... ", end = " ")
        data_norm = []
        size = self.grid_size 

        for mol in data:
            mol_norm = {}

            # Normalizo el Target
            Tn = (mol["T"] - fact["T"]["mean"]) / fact["T"]["std"]
            mol_norm["T"] = Tn

            # Normalizamos los atomos
            for key in ["H","C","N","O"]:
                nat = mol[key].shape[0]
                if  nat != 0:
                    ff = torch.zeros(nat, 4, size, size, size)
                    for ii in range(4):
                        ff[:, ii, :, :, :] = (mol[key][:, ii, :, :, :] - fact[key]["mean"][ii]) \
                            / fact[key]["std"][ii]

                else:
                    ff = torch.rand(0, 4, 7, 7, 7)
                
                ff = torch.nan_to_num(ff, nan = 0., posinf = 0., neginf = 0.)
                mol_norm[key] = ff
            
            # Agrego How Many
            mol_norm["how_many"] = mol["how_many"]

            # Genero el arreglo normalizado
            data_norm.append(mol_norm)

        print(str(np.round(time.time() - init, 2)) + " s.")
        return data_norm

    def _normalize_coeff(self, data, fact):
        init = time.time()
        print("Normalizando los datos... ", end = " ")
        data_norm = []

        for mol in data:
            mol_norm = {}
            for key in mol:
                if key != "how_many" and len(mol[key]) != 0:
                    mol_norm[key] = (mol[key] - fact[key]["mean"]) / fact[key]["std"]
                    mol_norm[key] = torch.nan_to_num(mol_norm[key], nan = 0., posinf = 0., neginf = 0.)
                elif key != "how_many" and len(mol[key]) == 0:
                    if key == "H":
                        mol_norm[key] = torch.rand(0, 2, 4)
                    else:
                        mol_norm[key] = torch.rand(0, 2, 34)
            
            # Agrego how many
            mol_norm["how_many"] = mol["how_many"]

            # Genero el arreglo normalizado
            data_norm.append(mol_norm)

        print(str(np.round(time.time(), 2)) + " s.")
        return data_norm 

    def _read_norm(self):
        if self.grid_size != False:
            if self.transform:
                name = self.path_dir + "factors_norm_trans.pickle"
            else:
                name = self.path_dir + "factors_norm_untr.pickle"
        elif self.coeff == True:
            name = self.path_dir + "factors_norm.pickle"
        else:
            print("Grid_size o Coeff debe elegirse como 'True'")
            exit(-1)

        try:
            with open(name,"rb") as f:
                factor_norm = pickle.load(f)
        except:
            print("No se pudo leer el", end=" ")
            print("archivo " + name)
            exit(-1)
        
        return factor_norm

# Modelos Atomicos
class Linear_Model(pl.LightningModule):
    def __init__(self, inp):
        super(Linear_Model, self).__init__()

        self.inp = inp
        # Arquitectura de la NN
        self.fc1 = nn.Linear(self.inp,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,1)

        self.act = nn.PReLU(num_parameters=1,init=0.25)

    def forward(self,x):
        if len(x) != 0:
            x = x.view(-1, self.inp)
            out = self.act(self.fc1(x))
            out = self.act(self.fc2(out))
            out = self.fc3(out)
        else:
            out = torch.tensor([]).type_as(x)

        return out

class Conv_Model(pl.LightningModule):
    def __init__(self):
        super(Conv_Model, self).__init__()

        # Arquitectura de la NN
        self.network = nn.Sequential(
            nn.Conv3d(4, 16, kernel_size=3, stride = 1, padding=1),
            nn.PReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride = 1, padding=1),
            nn.PReLU(),
            nn.MaxPool3d(2, 1), # output: 32 x 3 x 3 x 3

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.MaxPool3d(2, 2), # output: 64 x 2 x 2 x 2

            nn.Flatten(), 
            nn.Linear(64*3*3*3, 512),
            nn.PReLU(),
            nn.Linear(512, 64),
            nn.PReLU(),
            nn.Linear(64, 1))

    def forward(self,x):
        if len(x) != 0:
            out = self.network(x)
        else:
            out = torch.tensor([]).type_as(x)

        return out

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.PReLU()]
    if pool: layers.append(nn.MaxPool3d(2, 2))
    return nn.Sequential(*layers)

def linear_block(inpp, outt):
    layers = [nn.Linear(inpp, outt),
              nn.PReLU()]
    return nn.Sequential(*layers)

class Res_Model(pl.LightningModule):
    def __init__(self, inp):
        super(Res_Model, self).__init__()

        self.inp = inp

        # Arquitectura de la NN
        self.conv1 = linear_block(self.inp, 1024)
        self.conv2 = linear_block(1024, 512)
        self.res1 = nn.Sequential(linear_block(512, 512), linear_block(512, 512))

        self.conv3 = linear_block(512, 256)
        self.conv4 = linear_block(256, 128)
        self.res2 = nn.Sequential(linear_block(128, 128), linear_block(128, 128))

        self.lin = nn.Linear(128,1)
        
        #self.classifier = nn.Sequential(nn.MaxPool3d(2, 2), 
        #                                nn.Flatten(), 
        #                                nn.Dropout(0.2),
        #                                nn.Linear(128, 1))

    def forward(self,x):
        if len(x) != 0:
            x = x.view(-1, self.inp)
            out = self.conv1(x)
            out = self.conv2(out)
            out = self.res1(out) + out
            out = self.conv3(out)
            out = self.conv4(out)
            out = self.res2(out) + out
            #out = self.classifier(out)
            out = self.lin(out)
            return out
        else:
            out = torch.tensor([]).type_as(x)

        return out

# Definicion del Modelo
class Modelo(pl.LightningModule):
    def __init__(self,config):
        super().__init__()

        # Guardamos los hiperparametros
        self.save_hyperparameters(config)
        self.mod_type = self.hparams.Model
        self.size = self.hparams.grid_size
        self.batch_size = self.hparams.batch_size
        self.coeff = self.hparams.coeff
        
        # Instanciamos los modelos de cada elemento
        if self.mod_type == "Linear":
            self.Hydrogen  = Linear_Model(4)
            self.Carbon    = Linear_Model(34)
            self.Nitrogen  = Linear_Model(34)
            self.Oxygen    = Linear_Model(34)
        elif self.mod_type == "Conv" and self.coeff != True:
            self.Hydrogen  = Conv_Model()
            self.Carbon    = Conv_Model()
            self.Nitrogen  = Conv_Model()
            self.Oxygen    = Conv_Model()
        elif self.mod_type == "Res":
            self.Hydrogen  = Res_Model(4)
            self.Carbon    = Res_Model(34)
            self.Nitrogen  = Res_Model(34)
            self.Oxygen    = Res_Model(34)
        else:
            print("Solo se permiten redes neuronales feedforward (Linear),\
                convolucionales (Conv) o residuales (Res)")
            exit(-1)
        
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
        out   = torch.zeros(nmol,1).type_as(H)

        # Sumo los resultados
        # Hidrogeno
        if len(out_H) != 0:
            start = 0
            for mol in range(nmol):
                for at in range(int(start),int(start+Hw[mol,0].item())):
                    out[mol] += out_H[at]
                start += Hw[mol,0].item()
        
        # Carbono
        if len(out_C) != 0:
            start = 0
            for mol in range(nmol):
                for at in range(int(start),int(start+Hw[mol,1].item())):
                    out[mol] += out_C[at]
                start += Hw[mol,1].item()
        
        # Nitrogeno
        if len(out_N) != 0:
            start = 0
            for mol in range(nmol):
                for at in range(int(start),int(start+Hw[mol,2].item())):
                    out[mol] += out_N[at]
                start += Hw[mol,2].item()

        # Oxigeno
        if len(out_O) != 0:
            start = 0
            for mol in range(nmol):
                for at in range(int(start),int(start+Hw[mol,3].item())):
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
        self.log("train_loss",loss,on_epoch=True,on_step=False,prog_bar=False,
        batch_size = self.batch_size)

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
        self.log("val_loss",loss,on_epoch=True,on_step=False,prog_bar=False, 
        batch_size = self.batch_size)

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
                         patience=5, factor=0.5, verbose=True),
            "reduce_on_plateau": True,
            "monitor": "val_loss",
        }

        return [optim], [lr_scheduler]

    def graph_train(self):
        plt.title("Loss Function")
        plt.plot(self.train_loss,label="Training")
        plt.plot(self.val_loss,label="Validation")
        plt.yscale("log")
        plt.ylabel("MAE")
        plt.xlabel("Epoch")
        plt.legend()
        path = self.hparams.path_results
        plt.savefig(path + "/loss_train.png")

    def graph_test(self,fact):
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

        # Calculamos el MAE
        mae = np.abs(pred-real).mean()

        # Ajuste Lineal de los datos
        slope, intercept, r_value, p_value, std_err = \
                       stats.linregress(real,pred)

        # Graficamos los Datos
        title = "Energia predicha vs Real"
        etiqueta = "MAE " + str(np.round(mae,2)) + "kcal/mol"
        plt.title(title)
        plt.plot(real,pred,"o",label=etiqueta)
        plt.legend()

        # Graficamos la Recta
        r_value = np.round(r_value**2,2)
        etiqueta = "R^2: " + str(r_value)
        plt.plot(real, intercept + slope*real, 'r', label=etiqueta)
        plt.legend()

        plt.xlabel("Real [Kcal/mol]")
        plt.ylabel("Predicho [Kcal/mol]")
        
        # Guardamos la figura
        plt.savefig(path+"/pred_vs_real.png")

        # Guardamos los resultados
        np.savetxt(path+"/pred.txt",pred)
        np.savetxt(path+"/real.txt",real)

