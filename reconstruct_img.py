import matplotlib.pyplot as plt
from PIL import Image as im
import numpy as np

# input type: list[numpy]
def reconstruct_spectrogram(reconstructions, name=None):
    for i in range(len(reconstructions)):
        rec = reconstructions[i]
        if (np.min(rec) < 0):
            rec += abs(np.min(rec))
            times = 255 / np.max(rec)
            rec *= times
        img = im.fromarray(rec)
        img = img.convert('RGB')
        if name != None:
            img.save("./rec_imgs/"+str(name[i])+".png")
        else:
            img.save("./rec_imgs/"+str(i)+".png")

def latent_distrbution(latents_list, labels=None, name=""):
    if labels is None:
        labels = []
        for i in range(len(latents_list)):
            labels.append(i)
    for i, latent in enumerate(latents_list):
        latent = latent.detach().cpu().numpy()
        latent = latent.reshape((-1, 2))
        print(latent.shape)
        print(latent[:,0].shape, latent[:,1].shape)
        plt.scatter(latent[:,0],latent[:,1], s=3, alpha=0.5, labels=labels[i])
    plt.legend()
    plt.savefig("./rec_imgs/distrbution{}.png".format(name))