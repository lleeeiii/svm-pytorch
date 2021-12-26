import torch
import torch.nn as nn
import os.path
import matplotlib.pyplot as plt
import pandas as pd
from model import SVM, HingeLoss
from dataloader import Loader

def process_dataset(csv_file, input_cols, output_col, cat_one, cat_two):
    df = pd.read_csv(csv_file)
    # Filter columns on input and output columns only.
    df = df.filter(items = input_cols + [output_col])
    # Change category into 1 and -1
    df = df.loc[df[output_col].isin([cat_one, cat_two])]
    df[output_col+'Label'] = df[output_col].apply(lambda s: 1 if s == cat_one else -1)
    df = df.drop(output_col, axis=1)
    return df

def visualize_separator(model, df):
    model.eval()
    linear = model.linear
    w, b = linear.weight.data.flatten(), linear.bias.data
    w1, w2 = w
    # Draw the line determined by weight and bias.
    # Since w must be a nonzero vector, the point -bw/(|w|^2) must on the line.
    # The above point is also nearest to the origin from the line. 
    P1 = - b * w / (w1**2 + w2**2)
    P2 = P1 + torch.tensor([-w2, w1])
    plt.axline((P1[0], P1[1]), (P2[0], P2[1]))

    x1 = list(df[input_cols[0]])
    x2 = list(df[input_cols[1]])
    y = list(df[output_col+'Label'])
    color = ['g' if item == 1 else 'r' for item in y]
    plt.scatter(x1, x2, c=color)
    plt.xlim(min(x1)-0.1, max(x1)+0.1)
    plt.ylim(min(x2)-0.1, max(x2)+0.1)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

if __name__ == '__main__':
    # Input here
    inputs = [[5.1, 1.3], [4.6, 1.0], [6.0, 4.0]]
    input_dim = 2
    model_path = 'SVMmodel1.pth'
    draw_svm = True
    # Below 5 lines should be the same from train.py
    csv_file = "Iris.csv"
    input_cols = ['SepalLengthCm', 'PetalLengthCm']
    output_col = 'Species'
    cat_one = 'Iris-setosa'  # Will be labelled 1
    cat_two = 'Iris-versicolor' # Will be labelled -1

    # Pre-process dataset
    df = process_dataset(csv_file, input_cols, output_col, cat_one, cat_two)
    model = SVM(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Weight is {model.linear.weight.data}.")
    print(f"Bias is {model.linear.bias.data}.")
    print(f"The SVM straight line equation is ", )
    print("+".join(["("+str(round(model.linear.weight.data[0][i-1].item(),4))+")"+f"x{i}" for i in range(1, input_dim+1)]) + f"+({round(model.linear.bias.data.item(),4)})=0")
    # Use below to visualize Support Vector Machine if input_dim == 2
    if input_dim == 2 and draw_svm:
        visualize_separator(model, df)

    # Inference about input 
    input_tensor = torch.tensor(inputs).float()
    output = model(input_tensor)
    output = 2*(output >= 0).float()-1
    print(f"{len(inputs)} input(s) received.")
    for ind, input in enumerate(inputs):
        collect = [f"{input_cols[i]}: {input[i]}" for i in range(input_dim)]
        s = ", ".join(collect)
        pred = cat_one if output[ind] == 1 else cat_two
        s += f" => Prediction: {pred}."
        s = f"{ind+1}) " + s
        print(s)
