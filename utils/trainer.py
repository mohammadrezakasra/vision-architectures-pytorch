import os
import torch


class Trainer:
    """
    A unified training loop class for PyTorch models.
    Handles training, optional validation, AMP mixed precision, LR scheduling,
    metric logging, and checkpoint saving.
    """

    def __init__(self, model, optimizer, criterion, device, scheduler=None, save_dir="./checkpoints"):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.save_dir = save_dir

        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def _train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

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
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
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
            "train_loss": [],
            "train_acc": [],
            "lr": [],
        }

        if val_loader is not None:
            history["val_loss"] = []
            history["val_acc"] = []

        num_gpus = torch.cuda.device_count() if self.use_amp else 0
        print(f"Starting training on device: {self.device} (GPUs: {num_gpus}, AMP: {self.use_amp}) for {epochs} epochs...\n")

        best_acc = 0.0

        if model_name is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._train_epoch(train_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["lr"].append(current_lr)

            log_msg = (
                f"Epoch [{epoch:02d}/{epochs:02d}] | LR: {current_lr:.6f} | "
                f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}%"
            )

            current_acc = train_acc

            if val_loader is not None:
                val_loss, val_acc = self._evaluate_epoch(val_loader)

                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                log_msg += f" | Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%"
                current_acc = val_acc

            if current_acc > best_acc:
                best_acc = current_acc

                if model_name is not None:
                    save_path = os.path.join(self.save_dir, f"{model_name}.pth")
                    state_dict = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()
                    torch.save(state_dict, save_path)
                    log_msg += f" --> (New Best! Saved to {save_path})"

            print(log_msg)

        print(f"\nTraining Complete! Best Accuracy: {best_acc:.2f}%")
        return history