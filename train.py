import torch 
from INR import INR
from utils import *

import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(description="Train a model to predict RGB values from image coordinates.")
    
    parser.add_argument('image_path', type=str, help="Path to the input image.")
    parser.add_argument('--epochs', type=int, default=100000, help="Number of epochs to train the model.")
    parser.add_argument('--learning_rate', type=float, default=0.1, help="Learning rate for the optimizer.")
    parser.add_argument('--hidden_size', type=int, default=64, help="Size of the hidden layers.")
    
    args = parser.parse_args()
    
    model = INR(2, args.hidden_size, 1, 3).to(device)
    trained_model = train_model(args.image_path, model, epochs=args.epochs, learning_rate=args.learning_rate)
    
if __name__ == "__main__":
    main()