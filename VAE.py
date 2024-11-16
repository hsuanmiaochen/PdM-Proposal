import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn import metrics
from pathlib import Path
from typing import List, Dict, Optional
from voraus_ad import ANOMALY_CATEGORIES, Signals, load_torch_dataloaders
from configuration import Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH = Path.home() / "Downloads" / "voraus-ad-dataset-100hz.parquet"
MODEL_PATH: Optional[Path] = Path.cwd() / "model.pth"
configuration = Configuration(
    columns="machine",
    epochs=100,
    frequencyDivider=1,
    trainGain=1.0,
    seed=177,
    batchsize=8,#32
    nCouplingBlocks=4,
    clamp=1.2,
    learningRate=5.2e-4,
    normalize=True,
    pad=True,
    nHiddenLayers=0,
    scale=2,
    kernelSize1=13,
    dilation1=2,
    kernelSize2=1,
    dilation2=1,
    kernelSize3=1,
    dilation3=1,
    milestones=[11, 61],
    gamma=0.1,
)

# Define the VAE
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1):
        super(VAE, self).__init__()
        # Encoder
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)       # Mean
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)  # Log Variance

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)

    def encode(self, x):
        _, (hidden, _) = self.encoder_lstm(x)
        hidden = hidden[-1]  # Take the last hidden state
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        hidden_decoded = torch.relu(self.fc_dec(z)).unsqueeze(0)
        decoded, _ = self.decoder_lstm(hidden_decoded.repeat(1, 1, 1).permute(1, 0, 2))
        return decoded

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        decoded = self.decode(z)
        return decoded, mu, log_var

# VAE Loss Function
def vae_loss_function(recon_x, x, mu, log_var):
    # Reconstruction loss (MSE)
    recon_loss = nn.MSELoss()(recon_x, x)
    # KL divergence loss
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kld_loss / x.size(0)

# Training function for VAE
def train_vae() -> List[Dict]:
    train_dataset, _, train_dl, test_dl = load_torch_dataloaders(
        dataset=DATASET_PATH,
        batch_size=configuration.batchsize,
        columns=Signals.groups()[configuration.columns],
        seed=configuration.seed,
        frequency_divider=configuration.frequency_divider,
        train_gain=configuration.train_gain,
        normalize=configuration.normalize,
        pad=configuration.pad,
    )

    input_dim = train_dataset[0][0].shape[1]  # Number of features per timestep
    hidden_dim, latent_dim = 64, 16  # You may adjust these based on your dataset
    model = VAE(input_dim, hidden_dim, latent_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=5.2e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=configuration.milestones, gamma=configuration.gamma)

    training_results = []
    for epoch in range(configuration.epochs):
        model.train()
        total_loss = 0
        for data in train_dl:
            inputs, _ = data
            inputs = inputs.float().to(DEVICE)
            optimizer.zero_grad()
            outputs, mu, log_var = model(inputs)
            loss = vae_loss_function(outputs, inputs, mu, log_var)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch [{epoch+1}/{configuration.epochs}], Loss: {total_loss:.4f}')
        torch.save(model.state_dict(), MODEL_PATH)

        # Testing phase for AUROC calculation
        model.eval()
        result_list = []
        with torch.no_grad():
            for _, (tensors, labels) in enumerate(test_dl):
                inputs = tensors.float().to(DEVICE)
                outputs, _, _ = model(inputs)
                loss_per_sample = ((outputs - inputs) ** 2).mean(dim=(1, 2))  # MSE per sample
                for j in range(loss_per_sample.shape[0]):
                    result_labels = {k: v[j].item() if isinstance(v, torch.Tensor) else v[j] for k, v in labels.items()}
                    result_labels.update(score=loss_per_sample[j].item())
                    result_list.append(result_labels)

        # Calculate AUROC for each anomaly category
        results = pd.DataFrame(result_list)
        aurocs = []
        for category in ANOMALY_CATEGORIES:
            dfn = results[(results["category"] == category.name) | (~results["anomaly"])]
            fpr, tpr, _ = metrics.roc_curve(dfn["anomaly"], dfn["score"].values, pos_label=True)
            auroc = metrics.auc(fpr, tpr)
            aurocs.append(auroc)
            print(f'{category.name}, auroc={auroc:5.3f}')
        auroc_mean = np.mean(aurocs)
        training_results.append({"epoch": epoch, "aurocMean": auroc_mean, "loss": total_loss})

        print(f"Epoch {epoch:0>3d}: auroc(mean)={auroc_mean:5.3f}, loss={total_loss:.6f}")
        scheduler.step()

    return training_results

if __name__ == "__main__":
    train_vae()
