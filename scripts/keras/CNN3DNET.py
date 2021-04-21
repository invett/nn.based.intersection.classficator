import numpy as np
import torch.utils.data
import torch.nn as nn
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transf
import json
import matplotlib
import torchvision.models as models
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import argparse
from PIL import Image

def data_proc(datapath):
    
    with open(datapath, "r") as f:
         data = f.read().splitlines()
    
    sequences = [] 
    seq_player_id = eval(data[0])['player_id'] 
    sequence = [] 
    for frame in data: 
        frame = eval(frame) 
        player_id = frame['player_id']  
        if frame['player_id'] == seq_player_id: 
            if frame['velocity'] >= 0.0: 
                sequence.append({"id":frame["player_id"], "vehicle type":frame["player_type"], 'frame': frame['frame'], 'velocity': frame['velocity']}) 
        else: 
            seq_player_id = frame['player_id'] 
            sequences.append(sequence) 
            sequence = [] 
            if frame['velocity'] > 0.0: 
                sequence.append({"id":frame["player_id"], 'frame': frame['frame'], 'velocity': frame['velocity']}) 
                sequences.append(sequence) 
                
    normalized_seq=[]
    step=0
    seq_aux = []
    for i in sequences:
        if len(i) < 16:
            continue
        else:
            seq_aux = i[1::5]
            while len(seq_aux) < 16:
                seq_aux.append(seq_aux[-1])
            normalized_seq.append(seq_aux[0:16])
            seq_aux = []
    return normalized_seq

def set_divisor(p_train,p_val,normalized_seq):
    
    train_set = normalized_seq[0:int(p_train*len(normalized_seq))]
    val_set = normalized_seq[int(p_train*len(normalized_seq)):(int(p_train*len(normalized_seq))+int(p_val*len(normalized_seq)))]
    test_set = normalized_seq[int((p_train+p_val)*len(normalized_seq)):]    
    
    return train_set, val_set, test_set

def get_targets(train_set ,test_set ,val_set):
    trspeed=0
    tsspeed=0
    valspeed=0
    cont=0
    
    train_target = []
    test_target = []
    val_target = []
    
    for element in train_set:
        for image in element:
            trspeed += image["velocity"]
        #train_target.append((trspeed/len(element)-8.3333)/(27.7777-8.3333)) #normalización [0 - 1]
        train_target.append((trspeed/len(element)-(27.7777+8.3333)/2)/((27.7777-8.3333)/2)) #normalización [-1 - 1]
        trspeed = 0
        
    for element in test_set:
        for image in element:
            tsspeed += image["velocity"]
        #test_target.append((tsspeed/len(element)-8.3333)/(27.7777-8.3333)) #normalización [0 - 1]
        test_target.append((tsspeed/len(element)-(27.7777+8.3333)/2)/((27.7777-8.3333)/2)) #normalización [-1 - 1]
        tsspeed = 0
        
    for element in val_set:
        for image in element:
            valspeed += image["velocity"]
        #val_target.append((valspeed/len(element)-8.3333)/(27.7777-8.3333)) #normalización [0 - 1]
        val_target.append((valspeed/len(element)-(27.7777+8.3333)/2)/((27.7777-8.3333)/2)) #normalización [-1 - 1]
        valspeed = 0
    
    return train_target, test_target, val_target


class SimulatedMultaDataset(torch.utils.data.Dataset):
    def __init__(self, annotations, targets, path2, transforms=None) -> None:
        super().__init__()
        self.annotations = annotations
        self.targets = targets  # [30, 100] -> [-1, 1]
        self.transforms = transforms
        self.path=path2

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        images = [Image.open(self.path + "/{:015d}.png".format(sample["frame"])).convert('RGB') for sample in ann]
        if self.transforms:
            images = [self.transforms(image) for image in images] # 16 frames de [3, 112, 112]
        images = torch.stack(images, dim=1) # [16, 3, 112, 112]
        target = torch.as_tensor(self.targets[idx])
        return images, target

    def __len__(self):
        return len(self.annotations)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================
   
def main():
    # 1. Meter un buen parser de argumentos --> evitas "hardcodear" (todo parametrizado, nada puesto a manuja):
    # Por ejemplo: epochs = 10 está mal, es mejor meter un parametro max_epochs
    parser = argparse.ArgumentParser()
    parser.add_argument("--datalog", type=str,help="Path to json")
    parser.add_argument("--datafolder", type=str,help="Path to images")
    parser.add_argument("--max_epochs", default=10, type=int, help="Maximum number of arguments")
    parser.add_argument("--batch_size", type=int, default=5, help="Maximum number of arguments")
    parser.add_argument("--train_percentage",default=0.6, type=float, help="Percentage for training (0 to 1)")
    parser.add_argument("--val_percentage", default=0.201, type=float, help="Percentage for testing (0 to 1)")
    parser.add_argument("--pretrained", help="The model is pretrained?", action="store_true")
    parser.add_argument("--patience", type=int, default=7, help="patience of Early Stopping")
    args = parser.parse_args()
    pretrained_kinetics_mean = [0.43216, 0.394666, 0.37645]
    pretrained_kinetics_std = [0.22803, 0.22145, 0.216989]
    # 2. Obtener el modelo a partir de ellos
    model = models.video.r3d_18(pretrained=args.pretrained, progress=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.cuda()
    #layers = list(models.video.r3d_18(pretrained=args.pretrained, progress=False).children())[:-1]
    #fc_in_features = models.video.r3d_18().fc.in_features
    #layers.append(nn.Linear(fc_in_features, 1))
    #model = nn.Sequential(*layers).to("cuda")
    # 3. Obtener los "datasets" (de la clase SimulatedMultaDataset)
    normalized_seq = data_proc(args.datalog)
    train_set, val_set, test_set = set_divisor(args.train_percentage, args.val_percentage, normalized_seq)
    train_target, test_target, val_target = get_targets(train_set, test_set, val_set)
    train_transforms = transf.Compose([
        transf.Resize((112, 112)),
        transf.ToTensor(),
        # Normalize solo cuando el modelo está preentrenada
        #transf.Normalize(pretrained_kinetics_mean, pretrained_kinetics_std)
    ])
    test_transforms = transf.Compose([
        transf.Resize((112, 112)),
        transf.ToTensor(),
        # Normalize solo cuando el modelo está preentrenada
        #transf.Normalize(pretrained_kinetics_mean, pretrained_kinetics_std)
    ])
    val_transforms = transf.Compose([
        transf.Resize((112, 112)),
        transf.ToTensor(),
        # Normalize solo cuando el modelo está preentrenada
        #transf.Normalize(pretrained_kinetics_mean, pretrained_kinetics_std)
    ])
    train_dataset = SimulatedMultaDataset(train_set, train_target, args.datafolder, train_transforms)
    test_dataset = SimulatedMultaDataset(test_set, test_target, args.datafolder, test_transforms)
    val_dataset = SimulatedMultaDataset(val_set, val_target, args.datafolder, val_transforms)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=3, drop_last=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=3, drop_last=False)
    valid_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=3, drop_last=False)
    #   datasets = get_datasets(args.json_annotations_path, args.train_percentage, args.val_percentage)
    # 4. Una vez tenemos el dataset, toca el dataloader, que se puede separar como hace usted o bien se separan antes y se crean 3 datasets y 3 dataloaders (esto es lo más sencillo)
    # 5. Bucle de entrenamiento: podemos usar pytorch o PL . En estas primeras pruebas, vamos a hacerlo a la antigua usanza
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criteria = nn.MSELoss().cuda()
    
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    for epoch in range(args.max_epochs):
        print(f"Training epoch {epoch:04d}/{args.max_epochs:04d}")
        model.train()
        for i, batch in enumerate(train_dataloader):
            images, targets = batch
            images, targets = images.cuda(), targets.cuda()
            targets = torch.unsqueeze(targets, 1)
            optimizer.zero_grad()
            prediction = model(images)
            loss = criteria(prediction, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if i % 20 == 0:
                print(f"Batch {i:05d}, Loss: {train_losses[i]:.3f}")
        print(f"Validation epoch {epoch:04d}/{args.max_epochs:04d}")
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(valid_dataloader):
                images, targets = batch
                images, targets = images.cuda(), targets.cuda()
                targets = torch.unsqueeze(targets, 1)
                prediction = model(images)
                loss = criteria(prediction, targets)
                valid_losses.append(loss.item())
                if i % 20 == 0:
                    print(f"Batch {i:05d}, Loss: {valid_losses[i]:.3f}")

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        train_losses = []
        valid_losses = []

        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load('checkpoint2ndTry.pt'))
    real_target = []
    real_prediction = []
    episode_error = []
    
    abserror=0
    print("Test")
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            images, targets = batch
            images, targets = images.cuda(), targets.cuda()
            targets = torch.unsqueeze(targets, 1)
            prediction = model(images)
            loss = criteria(prediction, targets)
            print(f"Batch {i:05d}, Loss: {loss:.3f}\n")
            real_target.append(((targets.item()*(27.7777-8.3333)+27.7777+8.3333)/2)) 
            real_prediction.append(((prediction.item()*(27.7777-8.3333)+27.7777+8.3333)/2))
            abserror+=abs(real_target[i]-real_prediction[i])
            episode_error.append(abs(real_target[i]-real_prediction[i]))
            print(f"Target {real_target[i]:.3f}, Prediction: {real_prediction[i]:.3f}\n")
        print(f"Mean error: {abserror/len(real_prediction)} m/s\n")
        print(f"Mean error: {abserror/len(real_prediction)*3.6} Km/h\n")
    datos = np.zeros([122,2])
    
    #Guardamos los resultados del test en un np array.
    for i, element in enumerate(datos):
        element[0] = episode_error[i]
        element[1] = test_set[i][0]["id"]
        
if __name__ == '__main__':
    main()
    
