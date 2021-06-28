import torch as T
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import argparse
import matplotlib.pyplot as plt
from torchvision import utils
from tqdm import tqdm

device = T.device('cuda') if T.cuda.is_available() else T.device('cpu')

def make_net(env_p=3):
    if env_p == 3:
        cnn_out_dim = 256
    elif env_p == 4:
        cnn_out_dim = 768
    return nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(start_dim=1, end_dim=-1)
            ),
            nn.Sequential(
                nn.Linear(cnn_out_dim, 512),
                nn.ReLU()
            ),
            nn.Linear(512, 52)
        )
        
def make_dataloader(dataset, split=.9, batch_size=64, random_split=False):
    dataset_images = T.from_numpy(dataset["img"]).transpose(1,-1)#T.from_numpy(dataset["img"].astype(np.float32)/255.).transpose(1,-1)#.to(device)
    dataset_targets = T.from_numpy(dataset["obs"])#.to(device)
    if random_split:
        split_len = int((1.-split)*len(dataset_images))
        idx = np.random.randint(0, len(dataset_images)-split_len)
        train_dataset = TensorDataset(T.cat((dataset_images[:idx], dataset_images[idx+split_len:])),
                                        T.cat((dataset_targets[:idx], dataset_targets[idx+split_len:])))
        test_dataset = TensorDataset(dataset_images[idx:idx+split_len], dataset_targets[idx:idx+split_len])  
    else:
        idx = int(split*len(dataset_images))
        train_dataset = TensorDataset(dataset_images[:idx], dataset_targets[:idx])
        test_dataset = TensorDataset(dataset_images[idx:], dataset_targets[idx:])    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader, len(train_dataset), len(test_dataset)
    
def train_model(dataset, epochs, loss_save_loc="loss.png", batch_size=1024, lr=0.0003, random_split=False, verbose=False, net_path=None, env_p=3):
    if net_path:
        net = T.load(net_path).to(device)
    else:
        net = make_net(env_p=env_p).to(device)
    print("Net arch:", net)
    train_loader, test_loader, num_train, num_test = make_dataloader(dataset, batch_size=batch_size, random_split=random_split)
    print("Training number of samples:", num_train, "| number of batches:", len(train_loader))
    print("Evaluation number of samples:", num_test, "| number of batches:", len(test_loader))
    optimizer = optim.Adam(net.parameters(), lr=lr)
    training_losses = []
    eval_losses = []
    for e in tqdm(range(epochs)):
        if e % (epochs // 10) == 0:
            print("Epoch:", e, flush=True)
        net.train()
        training_loss = 0.
        for batch_idx, (data, target) in enumerate(train_loader):
            data = (data.float()/255.).to(device)
            target = target.to(device)
            output = net(data)
            loss = F.mse_loss(output, target)
            optimizer.zero_grad()
            loss.backward()
            training_loss += loss.item()
            optimizer.step()
        if verbose:
            print("Average loss during training epoch ", e, ": ", training_loss/num_train, sep="")
        training_losses.append(training_loss/num_train)
        net.eval()
        eval_loss = 0.
        for batch_idx, (data, target) in enumerate(test_loader):
            data = (data.float()/255.).to(device)
            target = target.to(device)
            output = net(data)
            eval_loss += F.mse_loss(output, target).item()
        if verbose:
            print("Average loss during eval epoch ", e, ": ", eval_loss/num_test, sep="")
        eval_losses.append(eval_loss/num_test)
    plt.title("Average loss per epoch")
    plt.plot(training_losses, label="Train")
    plt.plot(eval_losses, label="Eval")
    plt.ylim(ymin=0, ymax=eval_losses[0]*1.1)
    plt.legend()
    plt.savefig(loss_save_loc)
    return net.cpu()
    
def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    
def visualize_kernels(model, save_loc):
    filter = model[0][0].weight.data.clone().cpu()
    visTensor(filter)

    plt.axis('off')
    plt.ioff()
    plt.savefig(save_loc.replace('.', "_L0."))

    filter = model[0][2].weight.data.clone().cpu()
    visTensor(filter)

    plt.axis('off')
    plt.ioff()
    plt.savefig(save_loc.replace('.', "_L1."))

    filter = model[0][4].weight.data.clone().cpu()
    visTensor(filter)

    plt.axis('off')
    plt.ioff()
    plt.savefig(save_loc.replace('.', "_L2."))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="File path of dataset to train on.")
    parser.add_argument("--model_save_loc", type=str, default="model.pth", help="Path to save model to after training.")
    parser.add_argument("--figure_save_loc", type=str, default="loss.png", help="Path to save training/eval loss figure to.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size to use during training.")
    parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate to use during training.")
    parser.add_argument("--random_split", action="store_true", help="Indicates that dataset will be randomly split into test/eval (one split only done initially).")
    parser.add_argument("--verbose", action="store_true", help="Verbose output during training.")
    parser.add_argument("--net_path", type=str, default=None, help="(Optional) path to existing model to be trained.")
    parser.add_argument("--env_p", type=int, default=3, help="Value used for image-based environment will draw one in-game grid square as p^2 pixels.")
    parser.add_argument("--weight_viz_loc", type=str, default=None, help="(Optional) path to save weight visualization.")
    args = parser.parse_args()
    print(args, flush=True)
    
    dataset = np.load(args.dataset_path)
    trained_model = train_model(dataset, args.num_epochs, loss_save_loc=args.figure_save_loc, 
                                batch_size=args.batch_size, lr=args.learning_rate, random_split=args.random_split, 
                                verbose=args.verbose, net_path=args.net_path, env_p=args.env_p)
    T.save(trained_model.state_dict(), args.model_save_loc)
    T.save(trained_model[0].state_dict(), args.model_save_loc.replace(".", "_cnn."))
    T.save(trained_model[1].state_dict(), args.model_save_loc.replace(".", "_linear."))
    
    if args.weight_viz_loc:
        visualize_kernels(trained_model, args.weight_viz_loc)