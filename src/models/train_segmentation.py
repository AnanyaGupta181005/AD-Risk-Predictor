import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from src.models.unet import UNet
from pathlib import Path

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        # pos_weight=9.0 handles ~10% vessel pixel ratio
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9.0]))

    def forward(self, inputs, targets, smooth=1):
        # Flatten label and prediction tensors
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs_flat.sum() + targets_flat.sum() + smooth)  
        
        # Use simple BCE for stability in hybrid loss
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        return BCE + dice_loss

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def train_model(train_loader, val_loader, num_epochs=50, resume_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = DiceBCELoss().to(device)
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    start_epoch = 0
    history = {'train_loss': [], 'val_dice': []}
    best_dice = 0
    patience_counter = 0

    if resume_path and Path(resume_path).exists():
        print(f"Resuming from {resume_path}...")
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        history = checkpoint['history']
        best_dice = max(history['val_dice']) if history['val_dice'] else 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            with autocast(enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        # Validation Step
        model.eval()
        val_dice = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                # Simple Dice Score calculation for validation
                inter = (outputs * masks).sum()
                val_dice += (2. * inter) / (outputs.sum() + masks.sum() + 1e-6)
        
        avg_val_dice = (val_dice / len(val_loader)).item()
        history['train_loss'].append(epoch_loss / len(train_loader))
        history['val_dice'].append(avg_val_dice)

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Val Dice: {avg_val_dice:.4f}")

        # Checkpoint & Early Stopping
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            patience_counter = 0
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'history': history
            }, filename="best_model.pth.tar")
        else:
            patience_counter += 1

        if patience_counter >= 10:
            print("Early stopping triggered.")
            break

    return model, history