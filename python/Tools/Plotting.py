import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
import torch
import torch.nn.functional as F

def plot_image(image):
    image = torch.permute(image, (1,2,0))
    fig, axs = plt.subplots(1,1)
    axs.imshow(image)
    axs.set_title("camera image")
    axs.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)

def plot_images(images):
    L = torch.permute(images[:3,...],(1,2,0))
    R = torch.permute(images[3:,...],(1,2,0))
    diff = torch.abs(L - R)
    fig, axs = plt.subplots(1,3)
    axs[0].imshow(L)
    axs[0].set_title("left camera")
    axs[0].tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
    axs[1].imshow(R)
    axs[1].set_title("right camera")
    axs[1].tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
    axs[2].imshow(diff)
    axs[2].set_title("absolute diff")
    axs[2].tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)

def animate_run(obs_seq, title):
    images = torch.permute(obs_seq,(0,2,3,1)).cpu()
    images = torch.clamp(images, 0,1)

    xmax = images.shape[1]
    ymax = images.shape[2]

    fig, axs = plt.subplots(1,1)
    ax = plt.axes(xlim=(0,xmax), ylim=(ymax,0))
    im = plt.imshow(images[0])
    fig.suptitle(title)

    def animate(i):
        im.set_array(images[i])
        return [im]
    
    anim = animation.FuncAnimation(fig, animate, frames=images.shape[0], repeat=False)
    return anim



def animate_images(images, test_index):
    images = torch.permute(images,(0,2,3,1)).cpu()
    L = images[:,...,:3]
    R = images[:,...,3:]
    diff = torch.abs(L - R)

    fig = plt.figure()
    ax = plt.axes(xlim=(0,100), ylim=(100,0))
    im = plt.imshow(L[0])
    fig.suptitle("Input Image Seqeunce ID {0}".format(test_index))

    def animate(i):
        left = L[i]
        right= R[i]
        d = diff[i]

        im.set_array(left)
        return [im]

    anim = animation.FuncAnimation(fig, animate, frames=L.shape[0], repeat=False)
    return anim