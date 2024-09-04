import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image_as_coordinates_and_rgb(image_path):
    # Load the image using PIL
    image = Image.open(image_path)
    image = image.convert('RGB')  # Ensure it's in RGB format
    image = image.resize((256, 256))
    
    # Convert the image to a NumPy array and normalize RGB values to [0, 1]
    img_array = np.array(image) / 255.0
    print(img_array.shape)

    # Get the height and width of the image
    h, w, _ = img_array.shape
    
    # Create a meshgrid of coordinates
    x_coords, y_coords = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    
    # Flatten the coordinates and stack them
    coordinates = np.stack([x_coords.ravel(), y_coords.ravel()], axis=-1)
    
    # Flatten the image array to get corresponding RGB values
    rgb_values = img_array.reshape(-1, 3)
    
    # Convert to torch tensors
    coordinates = torch.tensor(coordinates, dtype=torch.float32)
    rgb_values = torch.tensor(rgb_values, dtype=torch.float32)
    
    return coordinates.to(device), rgb_values.to(device)

def train_model(image_path, model, epochs=100000, learning_rate=0.1):
    # Load the data
    coordinates, rgb_values = load_image_as_coordinates_and_rgb(image_path)
    
    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(coordinates)
        
        # Compute the loss
        loss = criterion(outputs, rgb_values)
        
        # Backward pass and optimize
        loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Print the loss
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    print("Training complete.")
    return model