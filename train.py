import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import os.path
import pandas as pd
import torch.optim as optim

from model import SVM, HingeLoss
from dataloader import Loader

################################################

# Input necessary data here
csv_file = "Iris.csv"
input_cols = ['SepalLengthCm', 'PetalLengthCm']
output_col = 'Species'
cat_one = 'Iris-setosa'  # Will be labelled 1
cat_two = 'Iris-versicolor' # Will be labelled -1
train_ratio = 0.8

epochs = 100
batch_size = 8
learning_rate = 0.05
weight_decay = 1e-3
visualize_data = False # Visualize only if input_cols has two entries
model_path = "SVMmodel1.pth"

################################################
# Preparing data
input_dim = len(input_cols)
print("Preparing data...")
if not os.path.isfile(csv_file):
    print("Couldn't find {}. Exit.".format(csv_file))
    exit()
df = pd.read_csv(csv_file)
# Filter columns on input and output columns only.
df = df.filter(items = input_cols + [output_col])
# Change category into 1 and -1
df = df.loc[df[output_col].isin([cat_one, cat_two])]
df[output_col+'Label'] = df[output_col].apply(lambda s: 1 if s == cat_one else -1)
input_train, input_val, output_train, output_val = train_test_split(df[input_cols], 
                                                df[output_col+'Label'], train_size=train_ratio, 
                                                stratify=df[output_col+'Label'])
train_dataset = Loader(input_train, output_train)
val_dataset = Loader(input_val, output_val)
print(f"Train dataset has {len(train_dataset)} entries.\nValidating dataset has {len(val_dataset)} entries.")
# Dump into Dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

################################################

# Visualize the data if input dimension is 2
if len(input_cols) == 2 and visualize_data:
    print("Visualizing data...")
    import matplotlib.pyplot as plt
    x1 = list(df[input_cols[0]])
    x2 = list(df[input_cols[1]])
    y = list(df[output_col+'Label'])
    color = ['g' if item == 1 else 'r' for item in y]
    plt.scatter(x1, x2, c=color)
    plt.show()

################################################
# Training
model = SVM(input_dim=input_dim)
criterion = HingeLoss
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

for epoch in range(epochs):  # loop over the dataset multiple times
    print("="*50)
    running_loss = 0.0
    train_loader_n = len(train_loader)
    for i, data in enumerate(train_loader, 0):
        model.train()
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        # print(f"Epoch {epoch+1} step {i+1}/{train_loader_n}, loss is {round(loss.item(), 2)}.")
        running_loss += loss.item()
    print(f"Epoch {epoch+1} loss is {round(running_loss/train_loader_n, 4)}.")

    eval_rloss = 0
    val_loader_n = len(val_loader)
    for i, data in enumerate(val_loader, 0):
        model.eval()
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # forward + backward + optimize
        outputs = model(inputs.float())
        loss = criterion(outputs, labels)

        # print statistics
        # print(f"Epoch {epoch+1} step {i+1}/{train_loader_n}, loss is {round(loss.item(), 2)}.")
        eval_rloss += loss.item()
    print(f"Epoch {epoch+1} eval loss is {round(eval_rloss/val_loader_n, 4)}.")

print('Finished Training')
print('Model parameters are')
print(model.linear.weight.data)
print(model.linear.bias.data)
torch.save(model.state_dict(), model_path)
print(f"Model saved in {model_path}.")