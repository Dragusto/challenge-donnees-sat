import torch.optim as optim
import wandb
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transforms

os.chdir('/tf/work/challenge-donnees-sat/')
from Quentin.framework.dataset  import CustomDataset, LandCoverData
from Quentin.framework.models  import NeuralNetwork
from Quentin.framework.utils  import AinterB, Data_augmentation
from Clement.frameworkPytorch.utils import compter_prob, compter_classes

#from frameworkPytorch.utils import compter_prob, compter_classes

def train_model_Unet(model, config, trainloader, testloader):
    if torch.cuda.is_available() : device= torch.device("cuda:0" )
        
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])

    n_epoch = config['n_epoch']
    if config['monitoring']:

        session = wandb.init(project='challenge-donnee-sat', entity='qchabennet',reinit=True)

        configwb = wandb.config
        configwb.lr = config['lr']
        configwb.batch_size = config['batch_size']
        configwb.model_type = config["model type"]
        configwb.nom_reseau = "Unet"
        configwb.optimizer_name = optimizer.__class__.__name__
        configwb.momentum = config['momentum']
        configwb.weighted = config['weighted']
        configwb.valeur_weight = config['valeur_weight']
        configwb.descriptif = config['descriptif']
    
    if config['weighted']:
        if config['test_snow'] == False :
            weights = torch.tensor(config['valeur_weight']).float()
        else : 
            weights = torch.tensor(LandCoverData.WEIGHT_CLASS_TEST_CLOUDS_SNOW).float()
    else:
        weights = torch.ones(10)

    if(torch.cuda.is_available()):
        model = model.to(device)
        weights = weights.to(device)
    
    loss = nn.CrossEntropyLoss(weight= weights)
    #boucle pour l'entrainement du réseau
    for epoch in range(n_epoch):  
        model.train()
        running_loss = 0.0
        t= tqdm(trainloader, desc="epoch %i" % (epoch+1),position = 0, leave=True)
        epoch_loop = enumerate(t)
        
        size = len(trainloader.dataset)
       
        for batch, dico in epoch_loop:
            images = dico['image']
            masques = dico['masque']
            if(torch.cuda.is_available()):
                images = images.to(device)
                masques = masques.to(device)
            # Compute prediction error
            pred = model(images)
            if config['test_snow'] == True :
                if batch%3==0:
                    images, masques = Data_augmentation(images, masques)
            # Backpropagation
            optimizer.zero_grad()
            loss_fn = loss(pred, masques)
            loss_fn.backward()
            optimizer.step()
            running_loss += loss_fn.item()
            
            if (batch+1) % config['freq monitoring'] == 0:
                t.set_description("epoch %i, 'mean loss: %.6f'" % (epoch+1,running_loss/config['freq monitoring']))
                t.refresh()
                wandb.log({"entropy_train" : running_loss/config['freq monitoring']})
                # calculer KL_train => faire une fonction qui part du masque prédit et du vrai masque et qui calcule la KL
                running_loss =0
            del images, masques, pred
        if config['monitoring']and epoch ==49:

            dic_log = test_model_Unet(model, config, testloader)
            dic_log["epoch"] = epoch+1
            wandb.log(dic_log)
            
    if config['test'] and epoch ==50:
        mask_cal = model_test_unet(model, config, trainloader)
        return mask_cal        


    session.finish()            
    print('Finished Training')
    
    
def test_model_Unet(model, config, dataloader):
    if torch.cuda.is_available() : device= torch.device("cuda:0" )
    last_batch = (len(dataloader.dataset)//config['batch_size'])+1
    dic_log = {}
    total_bonne_detect = np.zeros(10)
    total_pixel_class = np.ones(10)
    dic_res = {}
    epsilon = config['eps_KL']
    size = len(dataloader.dataset)
    mean_entropy = 0
    correct = 0
    probabilité = np.zeros(10)
    num_batches = len(dataloader)
    if config['weighted']:
        if config['test_snow'] == False :
            weights = torch.tensor(config['valeur_weight']).float()
        else : 
            weights = torch.tensor(LandCoverData.WEIGHT_CLASS_TEST_CLOUDS_SNOW).float()
    else:
        weights = torch.ones(10)
    if(torch.cuda.is_available()):
        model = model.to(device)
        weights = weights.to(device)
    
    loss = nn.CrossEntropyLoss(weight= weights)
    model.eval()
    KL=0
    test_loss = 0 
    correct = 0
    n_epoch =1
    for epoch in range(n_epoch):  
    
        running_loss = 0.0
        t= tqdm(dataloader, desc="batch %i" % (epoch+1),position = 0, leave=True)
        epoch_loop = enumerate(t)
        with torch.no_grad():
            for batch, dico in epoch_loop:

                images = dico['image']
                masques = dico['masque']
                mask = np.array(masques)
                if(torch.cuda.is_available()):
                    images = images.to(device)
                    masques = masques.to(device)
                pred = model(images)
                output = np.array(pred.detach().to('cpu'))
                masques_cal = np.argmax(output,axis=1)
                test_loss += loss(pred,masques)
                if(torch.cuda.is_available()):
                    correct += (pred.argmax(1)== masques).type(torch.float).sum().item()
                else:
                    correct += np.sum(np.ndarray.flatten(masques_cal) == np.ndarray.flatten(mask))
                batches = batch
                probabilité += AinterB(masques_cal,mask)
                for i in range(10):
                    a = np.ndarray.flatten(masques_cal) == i
                    b = np.ndarray.flatten(mask) == i
                    total_bonne_detect[i] += np.sum([all(tup) for tup in zip(a, b)])
                    total_pixel_class[i] += np.sum(b)
                    
                y_hat = np.array([compter_prob(mask) for mask in masques_cal])
                y_true = np.array([compter_prob(mask) for mask in mask])
                KL_by_class= ((y_true+epsilon)*np.log((y_true+epsilon)/(y_hat+epsilon)))
                KL += np.sum(np.sum(KL_by_class,axis =1))
                if ((batch+1) % config['freq monitoring'] == 0) or (batch+1 == last_batch):
                    if (batch+1 == last_batch):
                        accuracy_batch = (correct/(batches*config['batch_size']*masques.shape[1]*masques.shape[2]+(masques.shape[0]*masques.shape[1]*masques.shape[2])))*100
                    else: 
                        accuracy_batch = (correct/(((batches+1)*masques.shape[0]*masques.shape[1]*masques.shape[2])))*100
                    t.set_description("batch %i 'Accuracy_tot: %.4f'" % (batch, accuracy_batch))
                    t.refresh()

                del images, pred

    test_loss /= num_batches

    dic_res['accuracy'] = (correct/(size*masques.shape[1]*masques.shape[2]))*100
    print("accuracy_totale : ", dic_res['accuracy'])
    mean_KL = KL/size
    dic_res["mean_KL"] = round(mean_KL,3)
    mean_entropy= test_loss/(batches+1)
    dic_res['mean_entropy'] = np.round(np.array(mean_entropy.detach().to('cpu')),3)
    proba = 100*probabilité/(batch+(masques.shape[0]/config['batch_size']))
    print("proba class : ", proba)
    accuracy_class = np.round((total_bonne_detect/total_pixel_class)*100,1)
    for (categ, acc) in zip(LandCoverData.CLASSES[0:],accuracy_class[0:]):
         dic_res["acc_" + categ] = acc
            
    for (categ, iou) in zip(LandCoverData.CLASSES[0:],proba[0:]):
         dic_res["iou_" + categ] = iou
#     accuracy_class = ((probabilité/(batch+1))*100)
#     for (categ, acc) in zip(LandCoverData.CLASSES[1:],accuracy_class[1:]):
#          dic_res["acc_" + categ] = acc
    del masques
    print("dic_res : ",dic_res)
    return dic_res

def model_unet(model, config, dataloader):
    if torch.cuda.is_available() : device= torch.device("cuda:0" )
    last_batch = (len(dataloader.dataset)//config['batch_size'])+1
    dic = {}
    dic_res = {}
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_bonne_detect = np.zeros(10)
    total_pixel_class = np.zeros(10)
    probabilité = np.zeros(10)
    if config['weighted']:
        if config['test_snow'] == False :
            weights = torch.tensor(config['valeur_weight']).float()
        else : 
            weights = torch.tensor(LandCoverData.WEIGHT_CLASS_TEST_CLOUDS_SNOW).float()
    else:
        weights = torch.ones(10)
    if(torch.cuda.is_available()):
        model = model.to(device)
        weights = weights.to(device)
    epoch = 1

    model.eval()
    test_loss = 0 
    correct = 0
    n_epoch = 1
    for epoch in range(n_epoch):
        t= tqdm(dataloader, desc="epoch %i" % (epoch+1),position = 0, leave=True)
        epoch_loop = enumerate(t)
        with torch.no_grad():
            for batch, dico in epoch_loop:  
                images = dico['image']
                masques = dico['masque']
                mask = np.array(masques)
                ID = dico['id']
                if(torch.cuda.is_available()):
                    images = images.to(device)
                    masques = masques.to(device)
                pred = model(images)
                output = np.array(pred.detach().to('cpu'))
                masques_cal = np.argmax(output,axis=1)
                if(torch.cuda.is_available()):
                    correct += (pred.argmax(1)== masques).type(torch.float).sum().item()
                else:
                    correct += np.sum(np.ndarray.flatten(masques_cal) == np.ndarray.flatten(mask))
                probabilité += AinterB(masques_cal,mask, probabilité)
                batches = batch
                for i, test in enumerate(ID):
                    dic[test] = masques_cal[i,:,:]
                if ((batch+1) % config['freq monitoring'] == 0) or (batch+1 == last_batch):
                    accuracy_batch = (correct/(((batches+1)*masques.shape[0]*masques.shape[1]*masques.shape[2])))*100
                    t.set_description("batch %i 'Accuracy_tot: %.4f'" % (batch, accuracy_batch))
                    t.refresh()
                del images, pred
    

    dic_res['accuracy'] = (correct/(size*masques.shape[1]*masques.shape[2]))*100
    print("accuracy_totale : ",dic_res['accuracy'])
    dic_res['mean_entropy'] = np.round(np.array(mean_entropy.detach().to('cpu')),3)
    print("proba class : ", 100*probabilité/(batch+(masques.shape[0]/config['batch_size'])))
    accuracy_class = np.round((total_bonne_detect/total_pixel_class)*100,1)
    for (categ, acc) in zip(LandCoverData.CLASSES[1:],accuracy_class[1:]):
         dic_res["acc_" + categ] = acc
    for (categ, iou) in zip(LandCoverData.CLASSES[0:],probabilité[0:]):
         dic_res["iou_" + categ] = 100*probabilité/(batch+(masques.shape[0]/config['batch_size']))
#     accuracy_class = ((probabilité/(batch+1))*100)
#     for (categ, acc) in zip(LandCoverData.CLASSES[1:],accuracy_class[1:]):
#          dic_res["acc_" + categ] = acc
    del masques
    print(dic_res)
    return dic_res
def model_test_unet(model, config, dataloader):
    if torch.cuda.is_available() : device= torch.device("cuda:0" )
    dic = {}
    dic_res = {}
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    if(torch.cuda.is_available()):
        model = model.to(device)
    epoch = 1

    model.eval()
    test_loss = 0 
    correct = 0
    n_epoch = 1
    for epoch in range(n_epoch):
        t= tqdm(dataloader, desc="epoch %i" % (epoch+1),position = 0, leave=True)
        epoch_loop = enumerate(t)
        with torch.no_grad():
            for batch, dico in epoch_loop:  
                images = dico['image']
                ID = dico['id']
                if(torch.cuda.is_available()):
                    images = images.to(device)
                pred = model(images)
                batches = batch
                output = np.array(pred.detach().to('cpu'))
                masques_cal = np.argmax(output,axis=1)
                for i, test in enumerate(ID):
                    dic[test] = masques_cal[i,:,:]
                if (batch+1) %  config['freq monitoring'] == 0:
                    accuracy_batch = 0
                    t.set_description("batch %i 'Accuracy_tot: %.4f'" % (batch, accuracy_batch))
                    t.refresh()
                del images, pred
    
    
    return  dic
