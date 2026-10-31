import torch
import os


class Trainer:
    """
    A unified training loop class for PyTorch models.
    Handles training, optional validation, metric logging, and checkpoint saving.
    """

    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def _train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            # اصلاح سینتکس ضرب
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0
        return epoch_loss, epoch_acc

    def _evaluate_epoch(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0
        return epoch_loss, epoch_acc

    def fit(self, train_loader, val_loader=None, epochs=10, model_name=None):
        history = {
            'train_loss': [],
            'train_acc': []
        }

        if val_loader is not None:
            history['val_loss'] = []
            history['val_acc'] = []

        print(f"Starting training on device: {self.device} for {epochs} epochs...\n")

        best_acc = 0.0  # متغیر برای نگه‌داری بهترین دقت

        # ساخت پوشه برای ذخیره مدل‌ها در صورت نیاز
        if model_name is not None:
            os.makedirs("../checkpoints", exist_ok=True)

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._train_epoch(train_loader)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            log_msg = (
                f"Epoch [{epoch:02d}/{epochs:02d}] | "
                f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}%"
            )

            current_acc = train_acc  # به صورت پیش‌فرض ملاک را روی ترین می‌گذاریم

            if val_loader is not None:
                val_loss, val_acc = self._evaluate_epoch(val_loader)

                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                log_msg += f" | Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%"
                current_acc = val_acc  # اگر اعتبارسنجی داریم، ملاک ذخیره مدل را ولیدیشن قرار می‌دهیم

            # بررسی اینکه آیا رکورد جدیدی ثبت شده است یا خیر
            if current_acc > best_acc:
                best_acc = current_acc

                # ذخیره مدل در صورت ارسال نام مدل
                if model_name is not None:
                    save_path = f"../checkpoints/{model_name}.pth"
                    torch.save(self.model.state_dict(), save_path)
                    log_msg += f" --> (New Best! Saved to {save_path})"

            print(log_msg)

        print(f"\nTraining Complete! Best Accuracy: {best_acc:.2f}%")
        return history