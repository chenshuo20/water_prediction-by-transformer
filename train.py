import torch
import numpy as np
import models
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import argparse


def split_data(path, num_sequences, sequence_length, predict_length):
    df = pd.read_csv(path)
    values = df.values
    raw_data = torch.tensor(values, dtype=torch.float32)
    max_data = torch.max(raw_data, dim=0)[0]
    for i in range(raw_data.size()[1]):
        raw_data[:,i] = raw_data[:,i]/max(raw_data[:,i])
    step = (raw_data.size()[0]-sequence_length)//(num_sequences)
    data = []
    labels = []
    for i in range(num_sequences):
        data.append(raw_data[step*i:step*i+sequence_length])
        labels.append(raw_data[step*i+predict_length:step*i+sequence_length+predict_length])
    
    return torch.stack(data, dim=0), torch.stack(labels, dim=0), max_data


def train(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")   
    else:
        device = torch.device("cpu")  
    
    data, labels, maxs = split_data(args.data_path, args.num_sequences, args.sequence_length, args.predict_length)
    data.to(device), labels.to(device)
    # Split the data into training and validation sets
    cutting = int(args.training_set_ratio*args.num_sequences)
    train_data = data[:cutting]
    train_labels = labels[:cutting]
    val_data = data[cutting:]
    val_labels = labels[cutting:]
    # Initialize the model, optimizer, and loss function
    model = models.TransformerModel(args.input_dim, args.output_dim, args.num_layers, args.d_model, args.d_ff, args.dropout)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Train the model
    for epoch in range(args.num_epochs):
        total_loss = 0
        for i in tqdm(range(0, train_data.shape[0], args.batch_size)):
            optimizer.zero_grad()
            batch_data = train_data[i:i+args.batch_size].to(device)
            batch_labels = train_labels[i:i+args.batch_size].to(device)
            output = model(batch_data)
            output.to("cpu")
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('Epoch:', epoch+1, 'Loss:', total_loss/train_data.shape[0])

    # Evaluate the model on the validation set
    torch.save(model.state_dict(), 'model.pth')
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for i in range(0, val_data.shape[0], args.batch_size):
            batch_data = val_data[i:i+args.batch_size].to(device)
            batch_labels = val_labels[i:i+args.batch_size].to(device)
            output = model(batch_data)
            loss = criterion(output, batch_labels)
            total_loss += loss.item()
        print('Validation Loss:', total_loss/val_data.shape[0])


def get_args():

    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--num_layers', type=int, default=2,
                        help='the number of identical layers in the transformer encoder')
    parser.add_argument('--d_model', type=int, default=128,
                        help='the dimensionality of the hidden states in the transformer model')
    parser.add_argument('--d_ff', type=int, default=512,
                        help='dimensionality of the feedforward network')
    parser.add_argument('--dropout', type=int, default=0.1,
                        help='randomly drops out (sets to zero) some of the neuron activations in the neural network')
    parser.add_argument('--lr', type=int, default=0.0001,
                        help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of times the entire training dataset is passed through the learning algorithm during training')
    
    # training
    parser.add_argument('--training_set_ratio', type=float, default=0.8,
                        help='the ratio of training set')
    parser.add_argument('--num_sequences', type=int, default=1000,
                        help='the number of sequences sampled from the data')
    parser.add_argument('--sequence_length', type=int, default=100,
                        help='the length of sequences sampled from the data')
    parser.add_argument('--input_dim', type=int, default=4,
                        help='the input dimension of sequences')
    parser.add_argument('--output_dim', type=int, default=4,
                        help='the output dimension of sequences')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='the batchsize when training')
    parser.add_argument('--predict_length', type=int, default=12,
                        help='the predict length of the model')

    # other
    parser.add_argument('--seed', type=int, default=10,
                        help='random seed')
    parser.add_argument('--data_path', type=str, default='./Data.csv',
                        help='dataset path')
    parser.add_argument('--outdir', default='./outputs',
                        help='output data')
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    train(args)