import torch


def train_model(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        running_correct, running_total = 0, 0

        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)
            running_loss += loss.item()

        epoch_loss = running_loss / len(data_loader)
        epoch_accuracy = 100 * running_correct / running_total
        print(f"Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Accuracy={epoch_accuracy:.2f}%")

    return model