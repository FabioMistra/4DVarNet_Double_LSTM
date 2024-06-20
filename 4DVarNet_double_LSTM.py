## install libraries
import sys
import psutil
print(sys.executable)

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import functools as ft
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
import time
import copy
import itertools
import pandas as pd
import datamodule as dmod
from datamodule import compute_rmse, compute_re
# from model_AE_Bilin import model_AE
# from model_AE_Bilin2 import model_AE2
# from iter_solver import Model_4DVarNN_GradFP
from iter_solver2 import Model_4DVarNN_GradFP
# from iter_solver3 import Model_4DVarNN_GradFP
from model_AE_Bilin3D import model_AE as model_AE_3D



## Load the data
data_file  = {'model': 'DELFT3D_resized.nc', 'satellite': 'CMEMS_resized.nc', 'mask': 'landmask_resized.nc'}
model_data = xr.open_dataset(f'../../3_Data/{data_file["model"]}')
# sat_data   = xr.open_dataset(f'../../3_Data/{data_file["satellite"]}')
land_mask  = xr.open_dataset(f'../../3_Data/{data_file["mask"]}')

model_data  = model_data.isel(lon = slice(1, None))
# sat_data    = sat_data.isel(lon = slice(1, None))
land_mask   = land_mask.isel(lon = slice(1, None))

#### Adding a Gaussian noise 
# gauss_noise = np.random.normal(0,1, size= (model_data.SPM.shape))
# model_data['SPM'].data = model_data['SPM'].data + gauss_noise

##############################################################################################
## Initialize the dataset preprocessing
## new dataloader
input_da = dmod.load_bbp_data(GT=model_data, patch=model_data)

# Configuration parameters from base.yaml
config = {
    'input_da': input_da,
    'domains': {
        'train': {'time': slice('2016-01-01', '2016-12-31')},
        'val': {'time': slice('2017-01-01', '2017-06-30')},
        'test': {'time': slice('2017-07-01', '2018-01-01')}
    },
    'xrds_kw': {
        'patch_dims': {'time': 20, 'lat': len(model_data.lat.data), 'lon': len(model_data.lon.data)},
        'strides': {'time': 1, 'lat': len(model_data.lat.data), 'lon': len(model_data.lon.data)}
    },
    'dl_kw': {'batch_size': 1, 'num_workers': 1},
    'aug_factor': 1,
    'aug_only': True
}

# Instantiate the DataModule with the configuration
data_module = dmod.BaseDataModule(**config)

## Hyperparamters for the AE
DimAE = 64
downsamp = 2
dim_in = config['xrds_kw']['patch_dims']['time']
bilin_quad = False
total_RAM = torch.cuda.get_device_properties(0).total_memory / (1024**3)

## Load the AE model 
# ModelAE = model_AE(dim_in, DimAE, downsamp = downsamp, bilin_quad = bilin_quad)
# ModelAE2 = model_AE2(dim_in, DimAE, downsamp = downsamp, bilin_quad = bilin_quad)
ModelAE = model_AE_3D(dim_in = 1, DimAE = DimAE, downsamp = downsamp, bilin_quad = bilin_quad)
## Saving the hyperparameters values
hyperparam = f"time window: {dim_in} \nhidden layer dimension: {DimAE} \ndownsampling: {downsamp} \nbatch size: {config['dl_kw']['batch_size']} \nRAM memory: {total_RAM} GB"

with open("output_doubleLSTM.txt", "w") as file:
    # Write the output to the file
    file.write(hyperparam)
    file.write(f"\nbilin quad: {bilin_quad}")

## number of parameters
model_AE_param = f"\nnumber of parameters of the AE model: {sum(p.numel() for p in ModelAE.parameters() if p.requires_grad)}"
# model_AE_param = f"\nnumber of parameters of the AE model 2: {sum(p.numel() for p in ModelAE2.parameters() if p.requires_grad)}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ModelAE = ModelAE.to(device)
dev = "GPU" if torch.cuda.is_available() else "CPU"
dev_used = f"\nDevice used: {dev}"

with open("output_doubleLSTM.txt", "a") as file:
    # Write the output to the file
    file.write(model_AE_param)
    file.write(dev_used)


##############################################################################################
#setup the data module
data_module.setup()

meanTr, stdTr = data_module.norm_stats()

# Access the datasets
train_dataloader = data_module.train_dataloader()
val_dataloader = data_module.val_dataloader()
test_dataloader = data_module.test_dataloader()

## create the dataloader
dataloaders = {
    'train': train_dataloader,
    'val': val_dataloader
}

dataset_sizes = {'train': len(train_dataloader), 'val': len(val_dataloader)}


##############################################################################################
## MODEL LEARNING

UsePriodicBoundary = True 
InterpFlag         = False

tr_loss_list =[]
val_loss_list = []

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

patch_size = dim_in
batch_size = config['dl_kw']['batch_size']
lat = model_data['lat']
lon = model_data['lon']
shapeData = np.array((batch_size, patch_size, len(lat), len(lon)))
DimState = 96
alpha          = np.array([1.,0.1])

IterUpdate     = [0,100,200,500,2000,1000,1200]#[0,2,4,6,9,15]
NbProjection   = [0,0,0,0,0,0,0]#[0,0,0,0,0,0]#[5,5,5,5,5]##
NbGradIter     = [15,15]#[0,0,1,2,3,3]#[0,2,2,4,5,5]#
lrUpdate       = [1e-3,1e-4,1e-4,1e-5,1e-5,1e-4,1e-5,1e-6,1e-7]

NBGradCurrent   = NbGradIter[0]
NBProjCurrent   = NbProjection[0]
lrCurrent       = lrUpdate[0]

model           = Model_4DVarNN_GradFP(ModelAE,shapeData,DimState,NBProjCurrent,NBGradCurrent,UsePriodicBoundary)
model           = model.to(device)

full_model_param = f"\n4DVar model: Number of trainable parameters = {(sum(p.numel() for p in model.parameters() if p.requires_grad))}"

# fileAEModelInit = 'xxxx'

# optimization setting: freeze or not the AE
lambda_LRAE = 0.5
optimizer   = optim.Adam([{'params': model.model_Grad.parameters()},
                          {'params': model.model_Grad_input.parameters()},
                          {'params': model.model_AE.encoder.parameters(), 'lr': lambda_LRAE*lrCurrent},
                          ], lr=lrCurrent)

with open("output_doubleLSTM.txt", "a") as file:
    # Write the output to the file
    file.write(f"\nnumber of gradient iterations: {NbGradIter[0]} \nhidden layer dimension: {model.model_Grad.DimState} " )
    file.write(full_model_param)
    file.write("\n")




#################################################################################################################
## Main Loop 
# training function for dinAE
since = time.time()

alpha_Grad = alpha[0]
alpha_AE   = alpha[1]

best_model_wts = copy.deepcopy(model.state_dict())
best_loss = 1e10

num_epochs  = 50
comptUpdate = 1
iterInit    = 0
memory_usage = []

torch.autograd.set_detect_anomaly(True)

for epoch in range(iterInit,num_epochs):

    if ( epoch == IterUpdate[comptUpdate] ) & ( epoch > 0 ):
        # update GradFP parameters
        NBProjCurrent = NbProjection[comptUpdate]
        NBGradCurrent = NbGradIter[comptUpdate]
        lrCurrent     = lrUpdate[comptUpdate]

        if( (NBProjCurrent != NbProjection[comptUpdate-1]) | (NBGradCurrent != NbGradIter[comptUpdate-1]) ):
            # print("..... ")
            # print("..... ")
            # print("..... Update/initialize number of projections/Graditer in GradCOnvAE model # %d/%d"%(NbProjection[comptUpdate],NbGradIter[comptUpdate]))

            # update GradFP architectures
            # print('..... Update model architecture')
            # print("..... ")
            model = Model_4DVarNN_GradFP(model_AE,shapeData,NBProjCurrent,NBGradCurrent,UsePriodicBoundary)
            model = model.to(device)

            # copy model parameters from current model
            model.load_state_dict(best_model_wts)

            optimizer        = optim.Adam([{'params': model.model_Grad.parameters()},
                                    {'params': model.model_AE.encoder.parameters(), 'lr': lambda_LRAE*lrCurrent}
                                    ], lr=lrCurrent)

        else:
            # update optimizer learning rate
            # print('..... Update learning rate')
            mm = 0
            lr = np.array([lrCurrent,lambda_LRAE*lrCurrent])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr[mm]
                mm += 1

        # update counter
        if comptUpdate < len(IterUpdate)-1:
            comptUpdate += 1

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            torch.cuda.empty_cache()
            model.eval()

        running_loss         = 0.0
        running_loss_All     = 0.
        running_loss_AE      = 0.
        num_loss             = 0
        RMSE = 0.
        RE = 0.
        hidden = None
        cell   = None 
        
        # Iterate over data.
        for state, target in dataloaders[phase]:

            masks = torch.isnan(state).float()
            state = torch.nan_to_num(state)
            target = torch.nan_to_num(target)

            state      = state.to(device)
            masks      = masks.to(device)
            target     = target.to(device)
            inv_masks = 1. - masks

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # need to evaluate grad/backward during the evaluation and training phase for model_AE
            with torch.set_grad_enabled(True):
                state = torch.autograd.Variable(state, requires_grad=True)

                ## LSTM
                outputs,hidden_new,cell_new,normgrad = model(device, state,target,inv_masks,hidden,cell)

                ## New version using the previous hidden and cell in the LSTM

                loss_All    =  F.mse_loss(outputs, target)
                loss_AE     =  F.mse_loss(model.model_AE(outputs), outputs)
                loss_AE_GT  =  F.mse_loss(model.model_AE(target), target)
                RMSE_batch = compute_rmse(outputs, target, masks)
                RE_batch   = compute_re(outputs, target, masks)
                # print(loss_All.item(), loss_AE.item(), loss_AE_GT.item())

                loss  = alpha_Grad * loss_All + 0.5 * alpha_AE * ( loss_AE + loss_AE_GT )

                # backward + optimize only if in training phase
                if( phase == 'train' ):
                    loss.backward(retain_graph = True)
                    optimizer.step()
                
                hidden = torch.autograd.Variable(hidden_new, requires_grad = True)
                cell   = torch.autograd.Variable(cell_new, requires_grad = True)
                
                # statistics

            running_loss               += loss.item() * state.size(0)
            running_loss_All           += loss_All.item() * state.size(0)
            running_loss_AE            += loss_AE_GT.item() * state.size(0)
            num_loss                   += state.size(0)
            RMSE                       += RMSE_batch.item() * state.size(0)
            RE                         += RE_batch.item() * state.size(0)
            
            current_memory = torch.cuda.memory_allocated() / (1024**3)
            memory_usage.append(current_memory)

            del state, target, masks, inv_masks, outputs, hidden_new, cell_new

            torch.cuda.empty_cache() ## clear the memory


        epoch_loss       = running_loss / num_loss
        epoch_loss_All   = running_loss_All / num_loss
        epoch_loss_AE    = running_loss_AE / num_loss
        RMSE             = RMSE / num_loss
        RE               = RE / num_loss

        epoch_loss_All = epoch_loss_All * stdTr**2
        epoch_loss_AE  = epoch_loss_AE * stdTr**2


        if phase == 'train':
          tr_loss_list.append(epoch_loss)
        else:
          val_loss_list.append(epoch_loss)

        metrics = "\nepoch number {} {} Loss: {:.5e} RMSE: {:.5e} RE: {:.5e} LossAll: {:.5e} LossAE: {:.5e}" .format(epoch, phase, epoch_loss, RMSE, RE, epoch_loss_All, epoch_loss_AE)

        with open("output_doubleLSTM.txt", "a") as file:
            file.write(metrics)


        # deep copy the model
        if phase == 'val' and epoch_loss < best_loss:
            best_loss      = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

time_elapsed = time.time() - since

with open("output_doubleLSTM.txt", "a") as file:
    file.write( '\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60) )
    file.write( '\nBest val loss: {:4f}'.format(best_loss) )


#################
## Plotting the loss values through the epochs
plt.figure(1)
plt.plot(tr_loss_list, label="Train")
plt.plot(val_loss_list, label='Validation')
plt.xlabel("Epoch n.")
plt.ylabel("Loss value")
plt.legend(loc='best')

plt.savefig('epochs_loss_3D.png')

## Saving the best model
torch.save(best_model_wts, '../../5_Model_case_study/best_model_3D.pt')

plt.figure(2)
plt.plot(memory_usage, label = 'Memory Usage (GB)')
plt.ylabel('GB')
plt.legend(loc='best')

plt.savefig('memory_usage.png')