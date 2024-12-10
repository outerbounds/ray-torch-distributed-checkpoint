# Adapated from: https://docs.ray.io/en/latest/train/examples/pytorch/torch_fashion_mnist_example.html

import os
from typing import Dict
import time
import tempfile

import numpy as np
import torch
from filelock import FileLock
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Normalize, ToTensor

import ray
import ray.train
from ray import train
from ray.train.torch import TorchTrainer
from ray.train import (
    ScalingConfig,
    RunConfig,
    Checkpoint as Checkpoint,
    CheckpointConfig as CheckpointConfig,
)

BEST_CHECKPOINT_FILENAME = "best_model.pt"
LATEST_CHECKPOINT_FILENAME = "latest_model.pt"

def get_dataloaders(batch_size, val_only=False, as_ray_ds=False):

    def _preprocess_torch_dataset(torch_dataset):
        data = []
        for img, label in torch_dataset:
            data.append({"features": img.numpy(), "labels": int(label)})  # Single image and label
        return data

    transform = transforms.Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    if val_only:
        with FileLock(os.path.expanduser("~/data.lock")):
            test_data = datasets.FashionMNIST(
                root="~/data",
                train=False,
                download=True,
                transform=transform,
            )
            if as_ray_ds:
                data = _preprocess_torch_dataset(test_data)
                return ray.data.from_items(data)  # Individual items
            val_dataloader = DataLoader(test_data, batch_size=batch_size)
            return val_dataloader

    with FileLock(os.path.expanduser("~/data.lock")):
        training_data = datasets.FashionMNIST(
            root="~/data",
            train=True,
            download=True,
            transform=transform,
        )

        test_data = datasets.FashionMNIST(
            root="~/data",
            train=False,
            download=True,
            transform=transform,
        )

    if as_ray_ds:
        training_data = _preprocess_torch_dataset(training_data)
        testing_data = _preprocess_torch_dataset(test_data)
        return ray.data.from_items(training_data), ray.data.from_items(testing_data)

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(test_data, batch_size=batch_size)
    return train_dataloader, val_dataloader


def get_labels_map():
    return {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_func_per_worker(config: Dict):

    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size_per_worker"]
    try:
        checkpoint = config['checkpoint']
    except KeyError:
        checkpoint = None
    device = ray.train.torch.get_device()

    print(f"[my_ray_module] Preparing distributed data loaders...")
    train_dataloader, val_dataloader = get_dataloaders(batch_size=batch_size)
    train_dataloader = ray.train.torch.prepare_data_loader(train_dataloader)
    val_dataloader = ray.train.torch.prepare_data_loader(val_dataloader)

    model = NeuralNetwork()
    if checkpoint is not None:
        print(f"[my_ray_module] Resuming from checkpoint at {checkpoint.path}.")
        set_weights_from_checkpoint(model, checkpoint, device)
    model = ray.train.torch.prepare_model(model)
    print(f"[my_ray_module] Model on-device. Training model...")

    best_val_loss = float("inf")
    val_losses = []
    val_acc = []
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    t0_full = time.time()
    final_path = None
    for epoch in range(epochs):
        t0 = time.time()

        if ray.train.get_context().get_world_size() > 1:
            # Required for the distributed sampler to shuffle properly across epochs.
            train_dataloader.sampler.set_epoch(epoch)

        model.train()
        for X, y in train_dataloader:
            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss, num_correct, num_total = 0, 0, 0
        with torch.no_grad():
            for X, y in val_dataloader:
                pred = model(X)
                loss = loss_fn(pred, y)
                val_loss += loss.item()
                num_total += y.shape[0]
                num_correct += (pred.argmax(1) == y).sum().item()

        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)
        accuracy = num_correct / num_total
        val_acc.append(accuracy)

        rank = train.get_context().get_world_rank()
        checkpoint_dir = tempfile.mkdtemp()
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_losses": val_losses,
                "val_accuracy": val_acc,
            },
            os.path.join(checkpoint_dir, LATEST_CHECKPOINT_FILENAME),
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_losses": val_losses,
                    "val_accuracy": val_acc,
                },
                os.path.join(checkpoint_dir, BEST_CHECKPOINT_FILENAME),
            )
        ray_checkpoint = Checkpoint.from_directory(checkpoint_dir)
        ray.train.report(
            {"val_loss": val_loss, "accuracy": accuracy}, checkpoint=ray_checkpoint
        )

        tf = time.time()
        print(
            f"[my_ray_module] Model on-device. Last epoch took {round((tf-t0)/60, 3)} minutes. Training model..."
        )

    tf_full = time.time()
    print(f"[my_ray_module] Training completed in {round((tf_full-t0_full)/60, 3)} minutes!")


def train_fashion_mnist(
    num_workers=1,
    use_gpu=False,
    global_batch_size=32,
    learning_rate=1e-3,
    epochs=10,
    num_checkpoints_to_keep=2,
    checkpoint_storage_path=None,
    checkpoint=None
):

    train_config = {
        "lr": learning_rate,
        "epochs": epochs,
        "batch_size_per_worker": global_batch_size // num_workers,
    }
    if checkpoint is not None: 
        train_config['checkpoint'] = checkpoint

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(num_to_keep=num_checkpoints_to_keep),
        storage_path=checkpoint_storage_path,
        verbose=1,
    )
    scaling_config = ScalingConfig(
        num_workers=num_workers, 
        use_gpu=use_gpu,
    )
    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        run_config=run_config,
    )
    result = trainer.fit()
    return result

def set_weights_from_checkpoint(model_structure, checkpoint, device):
    with checkpoint.as_directory() as checkpoint_dir:
        checkpoint_dict = torch.load(
            os.path.join(checkpoint_dir, BEST_CHECKPOINT_FILENAME),
            map_location=device,
            weights_only=True,
        )
        state_dict = {
            k.replace("module.", ""): v
            for k, v in checkpoint_dict["model_state_dict"].items()
        }
        model_structure.load_state_dict(state_dict)

class TorchPredictor:
    
    def __init__(self, checkpoint: Checkpoint, cpu_only=False):
        self.device = torch.device("cpu") if cpu_only else torch.device("cuda")
        self.model = NeuralNetwork()
        set_weights_from_checkpoint(model_structure=self.model, checkpoint=checkpoint, device=self.device)
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        features = batch["features"]
        if features.ndim == 5 and features.shape[0] == 1:  # Case: (1, batch_sz, 1, 28, 28)
            features = features.squeeze(0)
        tensor = torch.as_tensor(features, dtype=torch.float32).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor).cpu().numpy().astype(np.float32)
            predicted_values = logits.argmax(axis=1)
        result = {"logits": logits, "predicted_values": predicted_values}
        return result


if __name__ == "__main__":
    train_fashion_mnist(num_workers=4, use_gpu=True)

