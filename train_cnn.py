import torch
import torch.nn as nn
import torch.optim as optim

from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model
from helper_lib.utils import save_model

device = (
    torch.device('mps') if torch.backends.mps.is_available()
    else torch.device('cuda') if torch.cuda.is_available()
    else torch.device('cpu')
)
print(f"Using device: {device}")

train_loader = get_data_loader('./data', batch_size=64, train=True)
test_loader = get_data_loader('./data', batch_size=64, train=False)

model = get_model("CNN")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

trained_model = train_model(model, train_loader, criterion, optimizer, device=device, epochs=10)

avg_loss, accuracy = evaluate_model(trained_model, test_loader, criterion, device=device)
print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

save_model(trained_model, 'cnn_model.pth')
print("Model saved to cnn_model.pth")

