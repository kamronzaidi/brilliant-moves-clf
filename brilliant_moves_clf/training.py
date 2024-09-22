import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
import argparse
import random
from shared.utility import *

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
            
def training_loop(Net, num_epochs, h1, h2, h3, dropout, lr, weight_decay, early_stop_limit, trainloader, testloader):
    train_loss, train_acc, test_loss, test_acc = [], [], [], []
    global device
    
    network = Net(h1,h2,h3,dropout).to(device)
    network.apply(reset_weights)

    optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc = 0
    best_epoch = 0
    best_net = None
    early_stop_count = 0
    
    for epoch in range(0, num_epochs):
        early_stop_count+=1
        if early_stop_limit is not None and early_stop_count > early_stop_limit:
            break

        current_loss = 0.0
        correct, total = 0, 0
        
        network.train()
        for k, data in enumerate(trainloader, 0):
            inputs, targets, sample_weights = [i.to(device) for i in data]
            optimizer.zero_grad()
            outputs = network(inputs.float())
            loss_function = nn.BCEWithLogitsLoss(weight=sample_weights)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            current_loss += loss.item()

            predicted = (outputs>0).float()
            total += sample_weights.sum().item()
            correct += ((predicted == targets)*sample_weights).sum().item()
        
        train_loss.append(current_loss/k)
        train_acc.append(correct/total)
    
        #Evaluation
        correct, total = 0, 0
        network.eval()
        with torch.no_grad():

            current_loss = 0
            for k, data in enumerate(testloader, 0):
                inputs, targets, sample_weights = [i.to(device) for i in data]
                outputs = network(inputs.float())

                predicted = (outputs>0).float()
                total += sample_weights.sum().item()
                correct += ((predicted == targets)*sample_weights).sum().item()
                
                loss_function = nn.BCEWithLogitsLoss(weight=sample_weights)
                loss = loss_function(outputs, targets)
                current_loss += loss.item()

            val_acc = correct / total
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                best_net = network
                early_stop_count = 0
                
            test_loss.append(current_loss/k)
            test_acc.append(val_acc)
    return network, best_net, best_epoch, best_acc, train_loss, train_acc, test_loss, test_acc

def CV_training_loop(train_dataset, Net, k_folds = 5, batch_size = 32, num_epochs = 200, early_stop_limit = 10, h1s = [10,25,50], h2s = [300,400,600], h3s = [10,25,50], dropout = 0.2, random_seed = 0, lr=1e-4, weight_decay=1e-5):
    results = []
    for h1 in h1s:
        results.append([])
        for h2 in h2s:
            results[-1].append([])
            for h3 in h3s:
                torch.manual_seed(random_seed)
                results[-1][-1].append({})
                print(f'-------- Model hyperparameters: h1={h1}, h2={h2}, h3={h3} --------')
                kfold = KFold(n_splits=k_folds, shuffle=True)

                # K-fold Cross Validation model evaluation
                for fold, (train_ids, test_ids) in enumerate(kfold.split(train_dataset)):
                    print(f'---------------- FOLD {fold} ----------------')
                    train_subsampler = SubsetRandomSampler(train_ids)
                    test_subsampler = SubsetRandomSampler(test_ids)

                    trainloader = DataLoader(
                                        train_dataset, 
                                        batch_size=batch_size, sampler=train_subsampler)
                    testloader = DataLoader(
                                        train_dataset,
                                        batch_size=batch_size, sampler=test_subsampler)

                    _, best_net, best_epoch, best_acc, _, _, _, _ = training_loop(Net, num_epochs, h1, h2, h3, dropout, lr, weight_decay, early_stop_limit, trainloader, testloader)
                    
                    save_path = f'models/model-fold-{fold}.pth'
                    torch.save(best_net.state_dict(), save_path)
                    results[-1][-1][-1][fold] = (best_epoch, best_acc)
                    
                print("---------------- Results ----------------")
                sum = 0.0
                epoch_sum = 0.0
                for key, value in results[-1][-1][-1].items():
                    print(f'Fold {key} | Best Epoch: {value[0]} | Best Accuracy: {round(value[1]*100)} %')
                    epoch_sum += value[0]
                    sum += value[1]*100
                print(f"Average best accuracy for hyperparameters ({h1,h2,h3}): {sum/len(results[-1][-1][-1].items())} | Average best epochs: {epoch_sum/len(results[-1][-1][-1].items())}")
                
def final_training_loop(train_dataset, test_dataset, Net, batch_size = 32, num_epochs = 21, early_stop_limit = None, h1=25, h2 = 400, h3 = 50, dropout = 0.2, random_seed = 0, lr=1e-4, weight_decay=1e-5):
    if random_seed is not None:
        torch.manual_seed(random_seed)
    print(f'-------- Final Model hyperparameters: h1={h1}, h2={h2}, h3={h3} --------')
    trainloader = DataLoader(
                        train_dataset, 
                        batch_size=batch_size)
    testloader = DataLoader(
                        test_dataset,
                        batch_size=batch_size)

    network, _, _, _, train_loss, train_acc, test_loss, test_acc = training_loop(Net, num_epochs, h1, h2, h3, dropout, lr, weight_decay, early_stop_limit, trainloader, testloader)
    
    save_path = f'models/model-final.pth'
    torch.save(network.state_dict(), save_path)
    print(f'Final Model | Train Accuracy: {round(train_acc[-1]*1000)/10}% | Test Accuracy: {round(test_acc[-1]*1000)/10}%')

def run_training(X, y, simple = True, class_balance = True, random_seed = 1, train_split = 0.9):
    if simple:
        y[y == 2] = 1
    ind = np.arange(X.shape[0])
    random.Random(random_seed).shuffle(ind)
    
    n_train = int(len(ind) * train_split)
    train_i = ind[:n_train]
    test_i = ind[n_train:]
    
    scaler = preprocessing.StandardScaler().fit(X[train_i])
    X = scaler.transform(X)
    
    X = torch.tensor(X)
    y = torch.tensor(y).view(-1, 1).float()
    train_i = torch.tensor(train_i).long()
    test_i = torch.tensor(test_i).long()

    sample_w = torch.ones(len(y.view(-1)))
    if class_balance:
        classes = (0,1) if simple else (0,1,2)
        for c in classes:
            sample_w[y.view(-1) == c] = len(y.view(-1))/(y.view(-1)==c).sum()/len(classes)
    sample_w = sample_w.view(-1, 1).float()
    
    train_dataset = TensorDataset(X[train_i], y[train_i], sample_w[train_i])
    test_dataset = TensorDataset(X[test_i], y[test_i], sample_w[test_i])
    
    CV_training_loop(train_dataset, NeuralNetworkDropout)
    final_training_loop(train_dataset, test_dataset, NeuralNetworkDropout)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'This script will train a neural network and save the state dict to a file')
    parser.add_argument("-d", "--moves_dir", help = "Path to directory containing the move folders.", default = 'moves')
    args = parser.parse_args()
    X, y = parse_trees(moves_dir=args.moves_dir, training=True)
    run_training(X, y)