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

def train_gan(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    generator, discriminator = model
    optimizer_g, optimizer_d = optimizer
    generator.to(device)
    discriminator.to(device)

    for epoch in range(epochs):
        for real_images, _ in data_loader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # === Discriminator ===
            noise = torch.randn(batch_size, 100, device=device)
            fake_images = generator(noise)

            d_real = discriminator(real_images)
            d_fake = discriminator(fake_images.detach())
            d_loss = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)

            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            # === Generator ===
            d_fake_for_g = discriminator(fake_images)
            g_loss = criterion(d_fake_for_g, real_labels)

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

        print(f"Epoch {epoch+1}/{epochs} done")

    return generator, discriminator



