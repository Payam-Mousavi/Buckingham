import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train_model(model, train_loader, val_loader, test_loader, optimizer, criterion, num_epochs, model_save_path, writer_path):
    best_val_loss = float('inf')
    best_model = None
    writer = SummaryWriter(writer_path)

    history = {
        'train_loss': [],
        'val_loss': [],
        'test_loss': None
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # Training loop
        for inputs, targets in train_loader:
            inputs, targets = inputs.float(), targets.float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        writer.add_scalar('Loss/Train', train_loss, epoch)
        history['train_loss'].append(train_loss)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.float(), targets.float()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model  # Save the model itself

        tqdm.write(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # Test loop
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.float(), targets.float()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)

    test_loss /= len(test_loader.dataset)
    writer.add_scalar('Loss/Test', test_loss, num_epochs)
    history['test_loss'] = test_loss

    tqdm.write(f'Final Test Loss: {test_loss:.4f}')

    # Save the best model
    if best_model is not None:
        torch.save(best_model, model_save_path)
    writer.close()

    return best_model, history