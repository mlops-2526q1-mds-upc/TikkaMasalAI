import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from PIL import Image
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
import io
from src.train.preprocess_data import create_data_loaders


class Food101Classifier(nn.Module):
    def __init__(self, num_classes=101, pretrained=True):
        super(Food101Classifier, self).__init__()
        # Use MobileNetV2 for efficiency (small and fast)
        self.backbone = mobilenet_v2(pretrained=pretrained)
        # Replace the classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(self.backbone.last_channel, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class Food101Trainer:
    def __init__(self, model, train_loader, val_loader, device="cuda"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)

        # Training history
        self.train_losses = []
        self.val_accuracies = []

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0

        print("before training")
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.6f}")

        avg_loss = running_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        self.val_accuracies.append(accuracy)
        return accuracy

    def train(self, epochs=5):
        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 30)

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_accuracy = self.validate()

            # Step scheduler
            self.scheduler.step()

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.2f}%")

        print("\nTraining completed!")
        return self.train_losses, self.val_accuracies

    def save_model(self, path="food101_model.pth"):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_losses": self.train_losses,
                "val_accuracies": self.val_accuracies,
            },
            path,
        )
        print(f"Model saved to {path}")


def main():
    # set up dev flag
    dev = True
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, num_classes = create_data_loaders(
        data_root="data/raw/food101/data", batch_size=64, dev=dev
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"Number of classes detected: {num_classes}")

    # Create model with detected number of classes
    print("Creating model...")
    model = Food101Classifier(num_classes=num_classes)

    # Create trainer
    trainer = Food101Trainer(model, train_loader, val_loader, device)

    # Train model
    train_losses, val_accuracies = trainer.train(epochs=5)

    # Save model
    trainer.save_model("models/food101_mobilenet.pth")

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()

    print(f"Best validation accuracy: {max(val_accuracies):.2f}%")


if __name__ == "__main__":
    main()
