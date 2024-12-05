import random
import numpy as np
import torch
import torch.distributions
from scipy.stats import mode
import os
import json

# Define directory to access image folders, labels, and pre-trained models
#drive.mount('/content/drive', force_remount = True)
#our_data_dir = './'

# Define the three possible data pathways
training_path = f'../data/data_training/'
evaluation_path = f'../data/data_evaluation/'
concept_path = f'../data/data_concept/'

# Store the sorted item names accordingly
training_tasks_files = sorted(os.listdir(training_path))
evaluation_tasks_files = sorted(os.listdir(evaluation_path))
concept_tasks_files = sorted(os.listdir(concept_path))

# Load the data into the objects X_test, y_test, X_train, y_train
focus = "evaluation" #  Define dataset to use for the rest of the script: "training", "evaluation", "concept"

# Load items from file list and open JSON files
focus_tasks = []
for task_file in eval(focus + "_tasks_files"):
    with open(f'{eval(focus + "_path")}{task_file}', 'r') as f:
        task = json.load(f)
        focus_tasks.append(task)


# Calculates how many pixels match between input and output
def accuracy(X_inp, X_out, exclude_zero=False):
    per_diff = []

    # Option to exclude the color zero for accuracy calculations
    if exclude_zero:
        for i in range(len(X_inp)):
            raw_diff = np.count_nonzero(np.logical_and(X_inp[i] == X_out[i], X_inp[i] != 0))
            (per_diff.append(raw_diff / np.count_nonzero(X_inp[i])) if np.count_nonzero(X_inp[i]) != 0 else per_diff.append(0))

    else:
        for i in range(len(X_inp)):
            raw_diff = np.count_nonzero(X_inp[i] == X_out[i])
            per_diff.append(raw_diff / X_inp[i].size)

    return per_diff

# Retrieve original input and reconstructed output for a model in evaluation mode.
def validate(model, eval_loader, X_inp, X_out, device):
    # Put model in evaluation mode and start reconstructions based on latent vector
    model.eval()
    with torch.no_grad():
        for batch_idx, (input, output) in enumerate(eval_loader):
            in_out = torch.cat((input, output), dim=0).to(device)
            z_out, _ = model.encode(in_out)
            out = model.decode(z_out)
            for i in range(len(in_out)):
                X_inp.append(reverse_one_hot_encoder(in_out[i].cpu().numpy()))
                X_out.append(reverse_one_hot_encoder(out[i].cpu().numpy()))
    return X_inp, X_out

# Print fully and partially solved items (100% vs. 95%)
def solved_tasks(X_inp, X_out):
    per_diff = accuracy(X_inp, X_out)

    full_comp = [i for i, e in enumerate(per_diff) if e == 1] # 100%
    n_full_comp = [i for i, e in enumerate(per_diff) if 1 > e > 0.95] # 95%

    print(f'The approach fully solved:')
    print(*full_comp, sep = ', ')
    print(f'The approach nearly solved:')
    print(*n_full_comp, sep = ', ')

    return full_comp

# Splits a list in half and returns each
def split_list(a_list):
    half = len(a_list)//2

    return a_list[:half], a_list[half:]

# Calculates the effect of convolution (amount, kernel, padding, stride) on the image dimensions (w x h)
def convo_eff(w = 30, num = 1, k = 3, p = 0, s = 1):
    for i in range(num):
        w = ((w - k + (2*p))/s) + 1

    return w

# Adds noise to the custom ARC format (empty cells = no color) for denoised Autoencoder
def add_noise(X, noise=0.3):
    X_noise = X[0].unsqueeze(0)
    for i in range(1, len(X)):
        # Clone tensor
        X_clone = X[i].detach().clone()

        # Count the number of 1s in the array
        num_ones = torch.count_nonzero(X_clone).item()

        # Calculate the number of 1s to replace with 0s
        num_to_replace = int(noise * num_ones)

        # Get the indices of the 1s
        indices = torch.argwhere(X_clone == 1).transpose(1,0)

        # Randomly shuffle the indices of the 1s
        idx = torch.randperm(indices[0].nelement())
        indices = indices[:, idx]

        # Replace the first num_to_replace 1s with 0s
        indices_to_replace = indices[:, :num_to_replace]
        X_clone[indices_to_replace[0,:], indices_to_replace[1,:], indices_to_replace[2,:]] = 0

        X_noise = torch.cat((X_noise, X_clone.unsqueeze(0)), dim=0)

    return X_noise

# Padding of the ARC matrices for convolutional processing
def padding(X, height=30, width=30, direction='norm'):
    h = X.shape[0]
    w = X.shape[1]

    a = (height - h) // 2
    aa = height - a - h

    b = (width - w) // 2
    bb = width - b - w

    if direction == 'norm':
        X_pad = np.pad(X, pad_width=((a, aa), (b, bb)), mode='constant')

    # Reverse padding for rescaling
    else:
        if height == 30 and width == 30:
            X_pad = X[:, :]
        elif height == 30:
            X_pad = X[:, abs(bb):b]
        elif width == 30:
            X_pad = X[abs(aa):a, :]
        else:
            X_pad = X[abs(aa):a, abs(bb):b]

    return X_pad

# Scaling of the ARC matrices using the Kronecker Product, retaining all the information
def scaling(X, height=30, width=30, direction='norm'):
    h = height/X.shape[0]
    w = width/X.shape[1]
    d = np.floor(min(h, w)).astype(int)

    X_scaled = np.kron(X, np.ones((d, d)))

    if direction == 'norm':
        return padding(X_scaled, height, width).astype(int)

    # Retain information for reverse scaling
    else:
        return d, X_scaled.shape

# Reverse scaling of the ARC matrices for final computations
def reverse_scaling(X_orig, X_pred):
    d, X_shape = scaling(X_orig, 30, 30, direction='rev') # get scaling information
    X_pad_rev = padding(X_pred, X_shape[0], X_shape[1], direction='rev') # reverse padding

    mm = X_shape[0] // d
    nn = X_shape[1] // d
    X_sca_rev = X_pad_rev[:mm*d, :nn*d].reshape(mm, d, nn, d)

    X_rev = np.zeros((mm, nn)).astype(int)
    for i in range(mm):
        for j in range(nn):
            X_rev[i,j] = mode(X_sca_rev[i,:,j,:], axis=None, keepdims=False)[0]

    return X_rev

# One-Hot-Encoding (i.e., dummy coding) of ARC matrices for 10 colors (w x h x color)
def one_hot_encoder(X):
    one_hot = (np.arange(10) == X[..., None]).astype(int)

    return np.transpose(one_hot, axes = [2,0,1])

# Reverse One-Hot-Encoding for easier visualization & final computations
def reverse_one_hot_encoder(X):
    return np.argmax(np.transpose(X, axes=[1,2,0]), axis=-1)

# Replace values in array with new ones from dictionary
def replace_values(X, dic):
    return np.array([dic.get(i, -1) for i in range(X.min(), X.max() + 1)])[X - X.min()]

# Convert ARC grids into numpy arrays
def flatten_dataset(X_full):
    X_fill = []
    for X_task in X_full:
        for X_single in X_task:
            X_fill.append(np.array(X_single))

    return X_fill

# Apply scaling, padding, and one-hot-encoding to arrays to get finalized grids
def get_final_matrix(X_full, stage="train"):
    if stage != "train":
        X_full = flatten_dataset(X_full)

    X_full_mat = []
    for i in range(len(X_full)):
        X_sca = scaling(X_full[i], 30, 30)
        X_one = one_hot_encoder(X_sca)
        X_full_mat.append(X_one)

    return X_full_mat

# Augment color of grids: randomly assign new colors to each color within a grid (creates 9 copies of original)
def augment_color(X_full, y_full):
    X_flip = []
    y_flip = []
    for X, y in zip(X_full, y_full):
        X_rep = np.tile(X, (10, 1, 1))
        X_flip.append(X_rep[0])
        y_rep = np.tile(y, (10, 1, 1))
        y_flip.append(y_rep[0])
        for i in range(1, len(X_rep)):
            rep = np.arange(10)
            orig = np.arange(10)
            np.random.shuffle(rep)
            dic = dict(zip(orig, rep))
            X_flip.append(replace_values(X_rep[i], dic))
            y_flip.append(replace_values(y_rep[i], dic))

    return X_flip, y_flip

# Augment orientation of grids: randomly rotates certain grids by 90, 180, or 270 degrees
def augment_rotate(X_full, y_full):
    X_rot = []
    y_rot = []
    for X, y in zip(X_full, y_full):
        k = random.randint(0, 4)
        X_rot.append(np.rot90(X, k))
        y_rot.append(np.rot90(y, k))

    return X_rot, y_rot

# Midpoint mirroring of grids: Creates copies of grids and mirrors at midpoint the left side
def augment_mirror(X_full, y_full):
    X_mir = []
    y_mir = []
    for X, y in zip(X_full, y_full):
        X_mir.append(X)
        y_mir.append(y)

        X_rep = X.copy()
        n = X_rep.shape[1]
        for i in range(n // 2):
            X_rep[:, n - i - 1] = X_rep[:, i]

        y_rep = y.copy()
        n = y_rep.shape[1]
        for i in range(n // 2):
            y_rep[:, n - i - 1] = y_rep[:, i]

        X_mir.append(X_rep)
        y_mir.append(y_rep)

    return X_mir, y_mir

# Combines array creation, augmentation, and preprocessing (e.g., scaling)
def preprocess_matrix(X_full, y_full, aug=[True, True, True]):
    X_full = flatten_dataset(X_full)
    y_full = flatten_dataset(y_full)

    if aug[0]:
        print("Augmentation: Random Color Flipping")
        X_full, y_full = augment_color(X_full, y_full)

    if aug[1]:
        print("Augmentation: Random Rotation")
        X_full, y_full = augment_rotate(X_full, y_full)

    if aug[2]:
        print("Augmentation: Midpoint Mirroring")
        X_full, y_full = augment_mirror(X_full, y_full)

    X_full = get_final_matrix(X_full)
    y_full = get_final_matrix(y_full)

    return X_full, y_full

