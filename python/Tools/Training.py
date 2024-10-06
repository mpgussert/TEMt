import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import Tensor, optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

import sys
import os

sys.path.append('../')
from TEMt.DataManagement import ImageSequencer, collator
from TEMt.Transformers import EndToEndTEMt
from TEMt.Modules.Resnet18VAE import VAE
from Tools.Plotting import plot_image, animate_run

##################################
# notebook run variables
code_dim = 256
chans = 3
rows = 64 # 64 hardcoded from loss.  i think i remember this being from the ResNet18?
cols = 64
imu_dim = 6
test_index = 100
batch_size = 100
num_steps = 25

Xtilde_dim = code_dim
Etilde_dim = 256
a_dim = 2
num_heads = 9


ID = str(code_dim)+ "_" + str(Etilde_dim)
root_path = "C:\\Users\\Horde\\Downloads\\Datasets"

load = True
model_name = "temt_50_epochs_"+ID
out_directory = "results_ETE_TEMt_" + ID + "_100_epochs"

if load:
    continued_directory = out_directory + "_continued"
else:
    continued_directory = out_directory

load_path = os.path.join(root_path, out_directory)
load_temt_path = os.path.join(load_path, model_name)
out_path = os.path.join(root_path, continued_directory)

temt_path = os.path.join(out_path, model_name)

try:
    os.makedirs(out_path)
except FileExistsError:
    print("directory exists")
    pass

files = ["C:\\Users\\Horde\\Downloads\\Datasets\\differential_walk_data_medium_1",
         "C:\\Users\\Horde\\Downloads\\Datasets\\differential_walk_data_large_1",
         "C:\\Users\\Horde\\Downloads\\Datasets\\random_walk_data_medium_1",
         "C:\\Users\\Horde\\Downloads\\Datasets\\random_walk_data_medium_2",
         "C:\\Users\\Horde\\Downloads\\Datasets\\random_walk_data_large_1"]

sequencer = ImageSequencer(files, image_rows=rows, image_cols=cols)

X, A, T = sequencer[test_index]
X = X.permute(0,2,3,1)
print(X.shape)

fig = plt.figure()
ax = plt.axes(xlim=(0,cols), ylim=(rows,0))
im = plt.imshow(X[0])
fig.suptitle("Raw input  {0}".format(test_index))
def animate(i):
    next_image = X[i]
    im.set_array(next_image)
    return [im]
anim = animation.FuncAnimation(fig, animate, frames=X.shape[0], repeat=False)
name = "X_test_index_{0}.gif".format(test_index)
anim.save(os.path.join(out_path, name))

plt.clf()
plt.close()

myTEMt = EndToEndTEMt(code_dim, a_dim, Xtilde_dim, Etilde_dim, num_heads).to(device)
if load:
    myTEMt.load_state_dict(torch.load(load_temt_path))
    myTEMt.train()

def get_results(index):
    with torch.no_grad():
        X, A, T = collator([sequencer[index]])
        X = X.float().to(device)
        A = A.float().to(device)
        T = T.float().to(device)
        
        print(X.shape, A.shape)
        Xpred, Xtilde, Etilde, _, _, _ = myTEMt(X.to(device), A.to(device))
        
        Xpred = Xpred.detach().cpu().squeeze()
        Xtilde = Xtilde.detach().cpu().squeeze()
        Etilde = Etilde.detach().cpu().squeeze()
        AttentionHeads = []
        for i in range(myTEMt._num_heads):
            AttentionHeads.append(myTEMt._multihead_attention.heads[i]._attention.squeeze().detach().cpu().numpy())
        X = X.detach().cpu().squeeze()
        A = A.detach().cpu().squeeze()
        
        return Xpred, Xtilde, Etilde, AttentionHeads, X, A, T


def get_inference_results(index):
    with torch.no_grad():
        X, A, T = sequencer[index]
        X = X.float().to(device)
        A = A.float().to(device)
        T = T.float().to(device)
        
        length = X.shape[0]
        diff = 100 - length
        fdiff = int(diff/2)
        bdiff = fdiff
        if diff%2 != 0:
            bdiff += 1

        xdim = X.shape[1:]
        adim = A.shape[1:]
        print(X.shape, A.shape)
        
        fPadX = torch.randn(fdiff, *xdim)/100.0
        bPadX = torch.randn(bdiff, *xdim)/100.0
        fPadA = torch.randn(fdiff, *adim)/100.0
        bPadA = torch.randn(bdiff, *adim)/100.0
        
        Xpadded = torch.clamp(torch.vstack((fPadX.to(device), X, bPadX.to(device))),0,1)
        Apadded = torch.vstack((fPadA.to(device), A, bPadA.to(device)))
        print(Xpadded.shape, Apadded.shape)
        Xpred, Xtilde, Etilde, X_decoded, X_code, X_logvars = myTEMt(Xpadded.unsqueeze(0), Apadded.unsqueeze(0))
        
        Xpred = Xpred[0,fdiff:-bdiff, ...].detach().cpu().squeeze()
        Xtilde = Xtilde[0,fdiff:-bdiff, ...].detach().cpu().squeeze()
        Etilde = Etilde[0,fdiff:-bdiff, ...].detach().cpu().squeeze()
        X_decoded = X_decoded[0,fdiff:-bdiff,...].detach().cpu().squeeze()
        X_code = X_code[0,fdiff:-bdiff,...].detach().cpu().squeeze()
        AttentionHeads = []
        for i in range(myTEMt._num_heads):
            AttentionHeads.append(myTEMt._multihead_attention.heads[i]._attention.squeeze().detach().cpu().numpy())
        X = X.detach().cpu().squeeze()
        A = A.detach().cpu().squeeze()
        
        return Xpred, Xtilde, Etilde, AttentionHeads, X,X_code, X_decoded, A, T

def rectangle(n):
    #get the two factors of N that will make the most squarish grid
    rootish = np.ceil(np.sqrt(n))
    sideish = np.floor(n/rootish)
    while sideish*rootish != float(n):
        rootish -= 1
        sideish = np.floor(n/rootish)
    return int(rootish), int(sideish)

Xpred0, Xtilde0, Etilde0, AttentionHeads0, X,X_code, X_decoded, A, T = get_inference_results(test_index)

plt.imshow(Xpred0)
plt.title("Predicted X codes, untrained")
plt.ylabel("time index")
print(torch.min(Xpred0),torch.max(Xpred0))
name = 'Xpred_untrained_E{0}X{1}H{2}_{3}.png'.format(Etilde_dim, Xtilde_dim, num_heads, ID)
plt.savefig(os.path.join(out_path, name))

plt.clf()
plt.close()

plt.imshow(Xtilde0)
plt.title("Xtilde codes, untrained")
plt.ylabel("time index")
print(torch.min(Xtilde0),torch.max(Xtilde0))
name = 'Xtilde_untrained_E{0}X{1}H{2}_{3}.png'.format(Etilde_dim, Xtilde_dim, num_heads, ID)
plt.savefig(os.path.join(out_path, name))

plt.clf()
plt.close()

plt.imshow(Etilde0)
plt.title("Etilde codes, untrained")
plt.ylabel("time index")
print(torch.min(Etilde0),torch.max(Etilde0))
name = 'Etilde_untrained_E{0}X{1}H{2}_{3}.png'.format(Etilde_dim, Xtilde_dim, num_heads, ID)
plt.savefig(os.path.join(out_path, name))

plt.clf()
plt.close()

headsx, headsy = rectangle(myTEMt._num_heads)
print(headsx, headsy)

fig, axs = plt.subplots(headsx,headsy, sharex=True, sharey=True)
if num_heads == 1:
    axs = [axs]
i = 0

if num_heads > 3:
    #axs[-1,:].set_xlabel("time_index") #things i wish i could do...
    for x in range(headsx):
        for y in range(headsy):
            # axs[x,y].set_title("head {0}".format(i))
            if y == 0:
                axs[x, y].set_ylabel("time index")
            if x == headsx-1:
                axs[x, y].set_xlabel("time index")
            axs[x,y].imshow(AttentionHeads0[i])
            i += 1
else:
    for i in range(num_heads):
        axs[i].imshow(AttentionHeads0[i])
fig.suptitle("Attention Heads, untrained")

plt.clf()
plt.close()

BATCH_SIZE = 20
alpha = 0

optimizer = optim.Adam(myTEMt.parameters(), lr=1E-4, weight_decay=1e-5)
# optimizer = optim.SGD(myTEMt.parameters(), lr=1E-4, weight_decay=1e-5)
mse = torch.nn.MSELoss(reduction='sum')
loader = DataLoader(sequencer, batch_size = BATCH_SIZE, collate_fn=collator, shuffle=True)

losses= []
Xpreds = []
Xtildes = []
Etildes = []
AttentionHeadses = []

for i in range(10):
    STEPS = 5
    for epoch in range(STEPS):
        total_loss = 0
        num_grads = 0
        for Xhost, Ahost, Thost in loader:
            # print("batched!")
            optimizer.zero_grad()
            X = Xhost.to(device)
            A = Ahost.to(device)
            # print(torch.max(X), torch.min(X))
            X_pred, Xtilde, Etilde, X_decoded, mu, logvar = myTEMt(X, A)
            # print(torch.max(X_decoded), torch.min(X_decoded))
            #print(X_decoded.shape, mu.shape, logvar.shape)
            #print(X_pred.shape, mu.shape)
            base = mse(X_pred[:,:-1], mu[:,1:])
            
            #reconstruction
            #Dkl = 0.5 * torch.sum(-1 - logvar + mu.pow(2) + logvar.exp())
            
            #compare sensorium reconstruction to input observation
            image_loss = F.binary_cross_entropy(X_decoded, X, reduction='sum')
            # norm_loss = torch.mean(torch.linalg.norm(mu, dim=0))*code_dim

            # print(norm_loss[0])
            #recon_loss =  image_loss + Dkl # + norm_loss
            
            #print(base.shape, regularization.shape)
            loss = image_loss + base
            loss.backward(retain_graph=True)
            total_loss += loss.item()
            optimizer.step()
        #total_loss.backward()
        Xpred, Xtilde, Etilde, AttentionHeads, _, _, _ = get_results(test_index)
        losses.append(total_loss)
        Xpreds.append(Xpred)
        Xtildes.append(Xtilde)
        Etildes.append(Etilde)
        AttentionHeadses.append(AttentionHeads)
        print(epoch, total_loss)
    #name = "temt_{0}_dims_{4}_epochs_{5}".format(code_dim, Xtilde_dim, Etilde_dim, num_heads, len(losses), ID)
    #torch.save(myTEMt.state_dict(),os.path.join(out_path, name))

    name = "temt_{0}_dims_{4}_epochs_{5}".format(code_dim, Xtilde_dim, Etilde_dim, num_heads, len(losses), ID)
    torch.save(myTEMt.state_dict(),os.path.join(out_path, name))

    plt.plot(losses)
    plt.xlabel("data epoch")
    plt.ylabel("loss")
    plt.title("MSE Loss vs Epoch")
    name = 'losses_{0}_E{1}X{2}H{3}_{4}.png'.format(len(losses), Etilde_dim, Xtilde_dim, num_heads, ID)
    plt.savefig(os.path.join(out_path, name))

    plt.clf()
    plt.close()

    Xpred, Xtilde, Etilde, AttentionHeads, X, X_code, X_decoded, A, T = get_inference_results(test_index)
    print(Xpred.shape)
    images_decoded = myTEMt._coder.decode(Xpred.to(device))
    images_decoded = images_decoded.cpu().detach().permute(0,2,3,1)
    X_decoded = X_decoded.cpu().detach().permute(0,2,3,1)
    print(images_decoded.shape)

    fig = plt.figure()
    ax = plt.axes(xlim=(0,64), ylim=(64,0))
    im = plt.imshow(images_decoded[0])
    fig.suptitle("Decoded Image Prediciton {0}".format(test_index))

    def animate(i):
        next_image = images_decoded[i]
        im.set_array(next_image)
        return [im]

    anim = animation.FuncAnimation(fig, animate, frames=Etilde.shape[0], repeat=False)
    name = "Xpred_{6}_E{0}X{1}H{2}_{3}frames_{4}_test_index_{5}.gif".format(Etilde_dim, Xtilde_dim, num_heads, Xpred.shape[0], ID, test_index, len(losses))
    anim.save(os.path.join(out_path, name))

    plt.clf()
    plt.close()

    fig = plt.figure()
    ax = plt.axes(xlim=(0,64), ylim=(64,0))
    im = plt.imshow(X_decoded[0])
    fig.suptitle("Decoded Image input {0}".format(test_index))

    def animate(i):
        next_image = X_decoded[i]
        im.set_array(next_image)
        return [im]

    anim = animation.FuncAnimation(fig, animate, frames=Etilde.shape[0], repeat=False)
    name = "X_decoded_{6}_E{0}X{1}H{2}_{3}frames_{4}_test_index_{5}.gif".format(Etilde_dim, Xtilde_dim, num_heads, Xpred.shape[0], ID, test_index, len(losses))
    anim.save(os.path.join(out_path, name))

    plt.clf()
    plt.close()

    plt.imshow(Xpred)
    plt.title("Predicted X codes, {0} epochs".format(len(losses)))
    plt.ylabel("time index")
    print(torch.min(Xpred), torch.max(Xpred))
    name = 'Xpred_{0}_E{1}X{2}H{3}_{4}.png'.format(len(losses), Etilde_dim, Xtilde_dim, num_heads, ID)
    plt.savefig(os.path.join(out_path, name))

    plt.clf()
    plt.close()

    plt.imshow(Xtilde)
    plt.title("Xtilde, {0} epochs".format(len(losses)))
    plt.ylabel("time index")
    name = 'Xtilde_{0}_E{1}X{2}H{3}_{4}.png'.format(len(losses), Etilde_dim, Xtilde_dim, num_heads, ID)
    plt.savefig(os.path.join(out_path, name))

    plt.clf()
    plt.close()

    plt.imshow(Etilde)
    plt.title("Etilde codes, {0} epochs".format(len(losses)))
    plt.ylabel("time index")
    name = 'Etilde_{0}_E{1}X{2}H{3}_{4}.png'.format(len(losses), Etilde_dim, Xtilde_dim, num_heads, ID)
    plt.savefig(os.path.join(out_path, name))
    print(torch.min(Etilde), torch.max(Etilde))

    plt.clf()
    plt.close()

    fig, axs = plt.subplots(headsx,headsy, sharex=True, sharey=True)
    if num_heads == 1:
        axs = [axs]
    i = 0

    if num_heads > 3:
        #axs[-1,:].set_xlabel("time_index") #things i wish i could do...
        for x in range(headsx):
            for y in range(headsy):
                # axs[x,y].set_title("head {0}".format(i))
                if y == 0:
                    axs[x, y].set_ylabel("time index")
                if x == headsx-1:
                    axs[x, y].set_xlabel("time index")
                axs[x,y].imshow(AttentionHeadses[-1][i])
                i += 1
    else:
        for i in range(num_heads):
            axs[i].imshow(AttentionHeads0[i])
    fig.suptitle("Attention Heads, untrained")

    plt.clf()
    plt.close()

    fig, axs = plt.subplots(headsx,headsy, sharex=True, sharey=True)
    if num_heads == 1:
        axs = [axs]
    i = 0
    artists = []
    if num_heads > 3:
        for x in range(headsx):
            row = []
            for y in range(headsy):
                # axs[x,y].set_title("head {0}".format(i))
                if y == 0:
                    axs[x, y].set_ylabel("time index")
                if x == headsx-1:
                    axs[x, y].set_xlabel("time index")
                row.append(axs[x,y].imshow(AttentionHeadses[0][i]))
                i += 1
            artists.append(row)
    else:
        for i in range(num_heads):
            artists.append(axs[i].imshow(AttentionHeadses[0][0]))

    def animate(i):
        heads = AttentionHeadses[i]
        j = 0
        if num_heads > 3:
            for x in range(headsx):
                for y in range(headsy):
                    # axs[x,y].set_title("head {0}".format(i))
                    if y == 0:
                        axs[x, y].set_ylabel("time index")
                    if x == headsx-1:
                        axs[x, y].set_xlabel("time index")
                    artists[x][y].set_data(heads[j])
                    j+=1
        else:
            for i in range(num_heads):
                artists[i].set_data(heads[i])
        fig.suptitle("Attention Heads, {0} epochs steps".format(len(losses)))
        return axs
        
    ani = animation.FuncAnimation(fig, animate, frames=len(AttentionHeadses), repeat=False)
    print(len(AttentionHeadses))

    name = "AttnHeads_{0}_E{1}X{2}H{3}_ImgSeq_{4}frames_{5}.gif".format(len(losses), Etilde_dim, Xtilde_dim, num_heads, Etilde.shape[0], ID)
    ani.save(os.path.join(out_path, name), fps = 10)

    plt.clf()
    plt.close()















