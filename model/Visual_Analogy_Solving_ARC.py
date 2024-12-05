
# Import standard library packages
# warnings.filterwarnings("ignore") # to reduce unnecessary output

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import torch
import torch.nn.functional as F
import torch.distributions
from numpy.linalg import norm
from scipy.stats.mstats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.stats.outliers_influence import variance_inflation_factor
from stepwise_regression.step_reg import forward_regression, backward_regression
from torch.optim import AdamW
import visualize
import utils
import math
import dataset
import models

plot_all = False

# Assign device used for script (e.g., model, processing)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"You are using a {device.type} device.")

# train: train (example) inputs (X) and output (y)
# test: test inputs (X) and output (y)
X_test, y_test, X_train, y_train = [[] for _ in range(4)]

# Distinguish between train (example) and test grids (input vs output)
for task in utils.focus_tasks:
    Xs_test, ys_test, Xs_train, ys_train = [[] for _ in range(4)]

    for pair in task["test"]:
        Xs_test.append(pair["input"])
        ys_test.append(pair["output"])

    for pair in task["train"]:
        Xs_train.append(pair["input"])
        ys_train.append(pair["output"])

    X_test.append(Xs_test)
    y_test.append(ys_test)
    X_train.append(Xs_train)
    y_train.append(ys_train)

# Reduce test inputs to one task for later model evaluation
test_item = 0 # e.g. ConceptARC test item No.2 would be here "1"
for i in range(len(X_test)):
    if len(X_test[i]) > 1:
        X_test[i] = [X_test[i][test_item]]
        y_test[i] = [y_test[i][test_item]]

# Example
if False or plot_all:
    print(X_test[3])
    visualize.summary_plot(X_test)

if False or plot_all:
    visualize.showcase_padding(dataset.data_load, X_train, y_train)

# Check convolutional effect on image/task size (for feature_dim adjustment)
print(f'convo_eff(): {utils.convo_eff(w = 30, num = 3, k = 4, p = 0, s = 2)}')

"""
Training the layers/weights of the VAE to generate representations allowing for accurate 
reconstructions of the given inputs. For each epoch the respective loss is printed, 
calculated through a combination of binary cross entropy and Kullback-Leibler divergence. 
The training data (X_train, y_train) is split using a validation split of 3:1, using 
the first split (N = 300) for training and the second split (N = 100) for evaluation 
purposes. The final model will be retrained on all training data points (N = 400) to 
preserve predictive power.
"""

# Define train-validation split
np.random.seed(5) # seed for getting same random samples
indices = np.random.permutation(len(X_train))
train_idx, val_idx = indices[:300].tolist(), indices[300:].tolist()

X_training, X_validation = [X_train[i] for i in train_idx], [X_train[i] for i in val_idx]
y_training, y_validation = [y_train[i] for i in train_idx], [y_train[i] for i in val_idx]


# Define model
vae = models.VariationalAutoencoder().to(device)
if os.path.isfile('models/model_128.pt'):
    vae.load_state_dict(torch.load('models/model_128.pt', weights_only=True, map_location=torch.device(device)))

# Load "training" data into PyTorch Framework
batch_size = 64
train_loader = dataset.data_load(X_training, y_training, aug=[True, True, True], batch_size=batch_size, shuffle=True)
test_loader = dataset.data_load(X_test, y_test)

if not os.path.exists('models'):
    os.makedirs('models')

if True or not os.path.isfile('models/model_128.pt'): # Train

    for r in range(1):
        def test_error(model, test_loader):
            model.eval()
            loss = 0
            with torch.no_grad():
                for batch_idx, (input, output) in enumerate(test_loader):
                    in_out = torch.cat((input, output), dim=0).to(device)
                    out, mu, logVar = model(in_out)
                    loss += F.binary_cross_entropy(out, in_out, reduction='sum')
            model.train()
            return loss.item()/len(test_loader)

        # Training the network for a given number of epochs
        def train(model, train_loader, epochs=50):
            optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.2)
            for epoch in range(epochs):
                model.train()
                train_loss = 0
                for batch_idx, (input, output) in enumerate(train_loader):

                    # Combine input & output, adding noise, attaching to device
                    in_out = torch.cat((input, output), dim=0)
                    in_out = in_out.to(device)

                    # Potential Denoised VAE variant; leave commented as this proved unfruitful
                    # in_out_noisy = add_noise(in_out, noise=0.2)
                    # in_out_noisy = in_out_noisy.to(device)

                    # Feeding a batch of images into the network to obtain the output image, mu, and logVar
                    out, mu, logVar = model(in_out)

                    # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
                    kl_divergence = -0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp())
                    bce = F.binary_cross_entropy(out, in_out, reduction='sum')
                    loss = bce + kl_divergence
                    train_loss += bce.item()

                    # Backpropagation based on the loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


                test_loss = test_error(model, test_loader)

                if epoch % 5 == 0:
                    print(f'Epoch: {epoch+1}, Loss: {train_loss/(len(train_loader)*batch_size):.1f}  '
                          f'test_loss: {test_loss:.1f}')
                    train_loss = 0

                if epoch % 100 == 0:
                    torch.save(model.state_dict(), 'models/model_128.pt')

            return model

        vae_final = train(vae, train_loader, epochs=100)

        torch.save(vae_final.state_dict(), 'models/model_128.pt')

        """
        Evaluating the above training through means of auxillary tools:
        1. Random display of input and respective reconstructions
        2. Plot demonstrating task reconstruction accuracy (measured by correct pixels)
        3. Heatmap illustrating individual pixel accuracy
        """

        # Load model for testing
        #model_vae = torch.load('models/model_128.pt', map_location=torch.device(device))
        model_vae = vae
        model_vae.load_state_dict(torch.load('models/model_128.pt', weights_only=True, map_location=torch.device(device)))
        model_vae.eval()

        # Load "validation" data into PyTorch Framework
        eval_loader = dataset.data_load(X_validation, y_validation)

    # Create lists to store input and output (reconstructions)
    X_inp, X_out = [], []

    def validate(model, eval_loader):
        # Put model in evaluation mode and start reconstructions based on latent vector
        model.eval()
        with torch.no_grad():
            for batch_idx, (input, output) in enumerate(eval_loader):
                in_out = torch.cat((input, output), dim=0).to(device)
                out, mu, logVar = model(in_out)
                for i in range(len(in_out)):
                    X_inp.append(utils.reverse_one_hot_encoder(in_out[i].cpu().numpy()))
                    X_out.append(utils.reverse_one_hot_encoder(out[i].cpu().numpy()))
        return X_inp, X_out

    X_inp, X_out = utils.validate(model_vae, eval_loader, X_inp, X_out, device)


    # Visualize five random tasks and their respective reconstructions (output)
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    # random.seed(4)
    for i in range(5):
        # idx = random.randrange(len(X_inp))
        visualize.plot_one(X_inp[i], axs[0,i], i, 'original input')
        visualize.plot_one(X_out[i], axs[1,i], i, 'reconstruction')

    # Plot differences between input and output matrices (reconstructions)
    visualize.plot_pix_acc(X_inp, X_out)

    # Plot heatmap of correct pixel determination by the model (absolute)
    visualize.plot_pix_heatmap(X_inp, X_out)


"""
# **Visual Analogy Solver (VAS)**

First, let's define the type of data (train vs. evaluation) we want to look at.
"""

# Define the data our VAS is looking at
y_obs = utils.flatten_dataset(y_test) # get the expected output
inp_index = np.insert(np.cumsum([len(i) for i in X_train]), 0, 0) # get the precise index of items

# Define loaders for train (example) and test data
test_loader_few = dataset.data_load(X_train, y_train)
test_loader_sol = dataset.data_load(X_test, y_test)

# Define lists of solved items per dataset
sol_training = [20, 37, 47, 52, 55, 102, 110, 114, 129, 177, 185, 222, 275, 290, 321, 333, 352, 354, 372, 398] # training data
sol_evaluation = [42, 88, 92, 148, 154, 171, 217, 271, 304, 307, 325, 328, 351, 363] # evaluation/test data
sol_concept = [11, 14, 17, 51, 52, 53, 54, 55, 57, 59, 74, 82, 100, 135] # conceptARC data

# What list to pick based on focus dataset
sol_full = eval("sol_" + utils.focus)

# Load model for model performance
#model_vae = torch.load('models/model_128.pt', weights_only=False, map_location=torch.device(device))
model_vae = vae
model_vae.load_state_dict(torch.load('models/model_128.pt', weights_only=True, map_location=torch.device(device)))
model_vae.eval()

"""
#%% md
### **Running the Model**

Next, we run the model (VAS). There are two modes to run the model:

1.   Fix latent representation to the mean of the probability distribution:

  `z_inp, _ = model.encode(input.to(device))`

2.   Sample the latent representation, producing (slightly) different solutions for the items in each run:

  `mu, logVar = model.encode(input.to(device))`

  `z_inp = model.reparameterize(mu, logVar)`

Depending on the chosen method, the other one has to be commented out. The below code demonstrates this. There are only three positions where this is necessary (`test_loader_few` input & output representation, `test_loader_sol` input representation).


"""

def visual_analogy(model, test_loader_few, test_loader_sol, inp_index, comp='average'):
    model.eval()
    with torch.no_grad():
        Z_few, Z_few_i, Z_sol, Z_sol_o, Z_sol_p = [[] for _ in range(5)]
        for batch_idx, (input, output) in enumerate(test_loader_few):

            z_inp, _ = model.encode(input.to(device))
            # mu, logVar = model.encode(input.to(device))
            # z_inp = model.reparameterize(mu, logVar)

            z_out, _ = model.encode(output.to(device))
            # mu, logVar = model.encode(output.to(device))
            # z_out = model.reparameterize(mu, logVar)

            z_diff = z_out.cpu().numpy().squeeze() - z_inp.cpu().numpy().squeeze()
            Z_few.append(z_diff)
            Z_few_i.append(z_inp.cpu().numpy().squeeze())

        for batch_idx, (input, output) in enumerate(test_loader_sol):

            z_inp, _ = model.encode(input.to(device))
            # mu, logVar = model.encode(input.to(device))
            # z_inp = model.reparameterize(mu, logVar)

            Z_sol.append(z_inp.cpu().numpy().squeeze())
            Z_sol_o.append(utils.reverse_one_hot_encoder(output.numpy().squeeze()))

        Z_avg, Z_sim, Z_cons, Z_rule = [[] for _ in range(4)]
        for i in range(len(inp_index)-1):

            # Average rule vector
            Z_avg.append(np.mean(Z_few[inp_index[i]:inp_index[i+1]], axis=0))

            # Similarity rule vector (euclidean distance)
            Z_temp = Z_few_i[inp_index[i]:inp_index[i+1]]
            Z_euc = []
            for t in range(len(Z_temp)):
                euclidean_sim = np.linalg.norm(Z_temp[t] - Z_sol[i])
                Z_euc.append(euclidean_sim)
            idx = list(range(inp_index[i], inp_index[i+1]))[Z_euc.index(min(Z_euc))]
            Z_sim.append(Z_few[idx])

            # Check encoding consistency of all rule vectors
            Z_temp = Z_few[inp_index[i]:inp_index[i+1]]
            Z_cons.append(cosine_similarity(Z_temp))

            # Check encoding consistency of two rule vector approaches
            t1 = (norm(Z_avg[i])*norm(Z_few[idx]))
            t2 = np.dot(Z_avg[i], Z_few[idx])
            if t1 == 0:
                t3 = 0.0
            else:
                t3 = t2 / t1

            #print(f"{i},   t2: {t2} / t1: {t1} = t3: {t3}")
            Z_rule.append(t3)

        Z_comp = Z_avg if comp == 'average' else Z_sim
        for i in range(len(Z_comp)):
            z_out = Z_sol[i] + Z_comp[i]
            out = model.decode(torch.tensor(z_out, dtype=torch.float32).unsqueeze(0).to(device))
            Z_sol_p.append(utils.reverse_one_hot_encoder(out.cpu().numpy()))

    return Z_sol_o, Z_sol_p, Z_cons, Z_rule


# While loop to keep running until something new got solved
sol = []
sol_full = [] # comment out if you want the model to continue searching (using the previously defined sol_full)
while all(i in sol_full for i in sol):
    Z_sol_o, Z_sol_p, Z_cons, Z_rule = visual_analogy(model_vae, test_loader_few, test_loader_sol, inp_index, comp='average')

    y_pred = []
    for i in range(len(y_obs)):
        y_pred.append(utils.reverse_scaling(y_obs[i], Z_sol_p[i]))

    sol = utils.solved_tasks(y_obs, y_pred)


"""
### **Display Predictions**

Next, we visualize the results and in the next chunk we may also investigate the accuracy of the model.
"""

visualize.plot_arc(325, path ='training')

# Choose the item to display the model's solution
idx = 74

# Plot ARC item
visualize.plot_arc(idx, path ='concept')

# Plot Original and Predicted Output (resized)
fig, axs = plt.subplots(2, 1, figsize=(6, 6))
visualize.plot_one(y_obs[idx], axs[0], 0, 'correct output')
visualize.plot_one(y_pred[idx], axs[1], 0, 'predicted output')

# Print Ruel Complexity
# print('Encoding Rule Consistency {0:.{1}f}:'.format(Z_rule[idx], 2))
# print(Z_cons[idx])


# Plot differences between original output and predicted output
visualize.plot_pix_acc(y_obs, y_pred, exclude_zero=False)

"""
## **Multiple Linear Regression**
Performing multiple regressions to investigate the impact of different task 
characteristics (e.g., Color Distribution, Background Coverage) on model accuracy. 
Accuracy is considered using both rule vector approaches (average vs. similarity) 
and output grid sizes (scaled vs. original).
"""

## Retrieve reconstruction accuarcy for the items in question
# Combine both lists to retrieve indexes for Reconstruction Accuracy
X_inp, X_out = [], []
X_full = [X + y for X, y in zip(X_train, y_train)]
inp_index = np.insert(np.cumsum([len(i) for i in X_full]), 0, 0)
# Create reconstructions for items in questions
rec_loader = dataset.data_load(X_train, y_train)
#X_inp, X_out = validate(model_vae, rec_loader)
X_inp, X_out = utils.validate(model_vae, rec_loader, X_inp, X_out, device)

# Calculate reconstruction accuarcy for items in questions
rec_diff = utils.accuracy(X_inp, X_out)
rec_avg = []
for i in range(len(inp_index)-1):
    rec_avg.append(np.mean(rec_diff[inp_index[i]:inp_index[i+1]], axis=0))

## Features for individual grids (input & output) - always calculated as the average per item
item_len = [len(i) for i in X_train] # No. of Examples
col_avg, void_avg, size_avg, cha_avg, sca_avg, size_diff = [[] for _ in range(6)]
for _train in [X_train, y_train]:
    for item in _train:
        col_per, void_per, size_per, ver_per, hor_per, sca_per = [[] for _ in range(6)]
        for i in item:
            col_per.append(len(np.unique(np.array(i))))
            void_per.append(np.count_nonzero(np.array(i) == 0) / np.array(i).size)
            size_per.append(np.array(i).size)
            ver_per.append(np.count_nonzero(np.diff(np.array(i), axis=0)))
            hor_per.append(np.count_nonzero(np.diff(np.array(i), axis=-1)))

            d, _ = utils.scaling(np.array(i), 30, 30, direction='rev')
            sca_per.append(d)

        col_avg.append(np.mean(col_per)) # Average colors (i.e., how many unique colors)
        void_avg.append(np.mean(void_per)) # Average void (i.e., how many zeroes)
        size_avg.append(np.mean(size_per)) # Average size (i.e., "size" or array)
        cha_avg.append(np.mean([sum(x) for x in zip(ver_per, hor_per)])) # Average Rate of Change
        sca_avg.append(np.mean(sca_per)) # Average scaling factor
        size_diff.append(1 if len(set(size_per)) > 1 else 0) # Binary; size differences between grids inputs

## Features for input to output changes - always calculated per item (e.g., average)
col_change, size_change, sim_avg = [[] for _ in range(3)]
for item_x, item_y in zip(X_train, y_train):
    x_sca , y_sca, xy_col, xy_size = [[] for _ in range(4)]
    for x, y in zip(item_x, item_y):
        xy_col.append(set(np.unique(np.array(x))) == set(np.unique(np.array(y))))
        xy_size.append(np.array(x).shape == np.array(y).shape)

        x_sca.append(utils.scaling(np.array(x), 30, 30))
        y_sca.append(utils.scaling(np.array(y), 30, 30))

    col_change.append(0 if all(xy_col) else 1) # Binary; color changes between input to output
    size_change.append(0 if all(xy_size) else 1) # Binary; size changes between input to output
    sim_avg.append(np.mean(utils.accuracy(x_sca, y_sca))) # Average similarity between input and output

## Features for example input to test input changes - always calculated per item
col_change_t, size_change_t = [[] for _ in range(2)]
for item_x, item_t in zip(X_train, X_test):
    flat_list = [i for first_list in item_x for sec_list in first_list for i in sec_list]
    col_full, col_test = set(flat_list), set(np.unique(np.array(item_t[0])))

    xy_size = []
    for x in item_x:
        xy_size.append(np.array(x).shape == np.array(item_t[0]).shape)

    col_change_t.append(0 if all(i in col_full for i in col_test) else 1) # Binary; color changes between input to input
    size_change_t.append(0 if any(xy_size) else 1) # Binary; size changes between input to input

## Dependent Variable: Accuracy (30x30 vs Original)
dep = "30" # or "original"
acc = utils.accuracy(Z_sol_o, Z_sol_p) if dep != "original" else utils.accuracy(y_obs, y_pred) # last accuracy() is "original" size

## Split concatenated features from above (containing input and output features) into halves
feature_split = {}
for feature in ["col_avg", "void_avg", "size_avg", "cha_avg", "sca_avg", "size_diff"]:
    feature_split[f'{feature}_x'], feature_split[f'{feature}_y'] = utils.split_list(eval(feature))

## Create final DataFrame
(data := pd.DataFrame({'Number_Examples': item_len,
                       'Average_Colors_X': feature_split['col_avg_x'],
                       'Average_Colors_Y': feature_split['col_avg_y'],
                       'Average_Zeros_X': feature_split['void_avg_x'],
                       'Average_Zeros_Y': feature_split['void_avg_y'],
                       'Average_Size_X': feature_split['size_avg_x'],
                       'Average_Size_Y': feature_split['size_avg_y'],
                       'Average_RoC_X': feature_split['cha_avg_x'],
                       'Average_RoC_Y': feature_split['cha_avg_y'],
                       'Average_Scale_X': feature_split['sca_avg_x'],
                       'Average_Scale_Y': feature_split['sca_avg_y'],
                       'Average_Reconstruction': rec_avg,
                       'Average_Similarity': sim_avg,
                       'Size_Differences': feature_split['size_diff_x'],
                       'Color_Change': col_change,
                       'Color_Change_T': col_change_t,
                       'Grid_Size_Change': size_change,
                       'Grid_Size_Change_T': size_change_t,
                       'Accuracy': acc}))


# Define predictors and dependent variable (standardized)
X = zscore(data.iloc[:, 0:-1])
y = zscore(data['Accuracy'])

# Check for Multicollinarity
vif_data = pd.DataFrame({"Feature": X.columns,
                         "VIF": [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]})

# Fit model with standardized features
estimates = sm.OLS(y, sm.add_constant(X)).fit()

# Run LASSO as additional feature selection (to stepwise & RFECV)
lasso = sm.OLS(y, X).fit_regularized(alpha=0.1, L1_wt=1)

# Run cross-validated Recursive Feature Elimination (RFE)
rfe_mod =  RFECV(LinearRegression(), min_features_to_select=5, cv=300)
rfe_ranked = pd.DataFrame({"Feature": X.columns,
                           "Rank": rfe_mod.fit(X, y).ranking_})

print(estimates.summary())
print('==============================================================================\n          Feature Selection\n')
print('Features based on forward stepwise regression:')
print(*forward_regression(X, y, threshold_in=0.01), sep = ', ')

print('\nFeatures based on backward stepwise regression:')
print(*backward_regression(X, y, threshold_out=0.01), sep = ', ')

print('\nFeatures based on cross-validated recursive feature elimination:')
print(*rfe_ranked[rfe_ranked['Rank']==1]['Feature'].tolist(), sep = ', ')

print('==============================================================================\n               Lasso\n', lasso.params)
# print('==============================================================================\n          Multicollinearity\n', vif_data.to_string(index=False))


visualize.do_scatter_plot(data)

"""
## **Appendix**

Solved ConceptARC items based on rule vector approach. For every concept category, we calculated the solved fraction of items (out of 30).
"""
# Average Rule Vector
# [11, 14, 51, 52, 53, 54, 55, 59, 100, 135] on 1st test input
# [22, 23, 40, 51, 52, 54, 55, 57, 59, 74, 78, 100] on 2nd test input
# [51, 52, 55, 59, 82, 100, 121, 138] on 3rd test input

# 'AboveBelow (0-9)': 0
# 'Center' (10-19): 2/30 = 0.07
# 'CleanUp' (20-29): 2/30 = 0.07
# 'CompleteShape' (30-39): 0
# 'Copy' (40-49): 1/30 = 0.03
# 'Count' (50-59): 16/30 = 0.53
# 'ExtendToBoundary' (60-69): 0
# 'ExtractObjects' (70-79): 2/30 = 0.07
# 'FilledNotFilled' (80-89): 1/30 = 0.03
# 'HorizontalVertical' (90-99): 0
# 'InsideOutside' (100-109): 3/30 = 0.1
# 'MoveToBoundary' (110-119): 0
# 'Order' (120-129): 1/30 = 0.03
# 'SameDifferent' (130-139): 2/30 = 0.07
# 'TopBottom2D' (140-149): 0
# 'TopBottom3D' (150-159): 0


# Similarity Rule Vector
# [14, 51, 52, 53, 54, 55, 59, 147] on 1st test input
# [23, 51, 52, 53, 54, 55, 59, 74, 78, 100, 135, 140] on 2nd test input
# [51, 52, 53, 54, 55, 59, 82, 138] on 3rd test input

# 'AboveBelow (0-9)': 0
# 'Center' (10-19): 1/30 = 0.03
# 'CleanUp' (20-29): 1/30 = 0.03
# 'CompleteShape' (30-39): 0
# 'Copy' (40-49): 0
# 'Count' (50-59): 18/30 = 0.6
# 'ExtendToBoundary' (60-69): 0
# 'ExtractObjects' (70-79): 2/30 = 0.07
# 'FilledNotFilled' (80-89): 1/30 = 0.03
# 'HorizontalVertical' (90-99): 0
# 'InsideOutside' (100-109): 1/30 = 0.03
# 'MoveToBoundary' (110-119): 0
# 'Order' (120-129): 0
# 'SameDifferent' (130-139): 2/30 = 0.07
# 'TopBottom2D' (140-149): 2/30 = 0.07
# 'TopBottom3D' (150-159): 0

