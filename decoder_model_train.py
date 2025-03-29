import torch
import time
import math
import torch.nn as nn
import torch.optim as optim
from dataloader import get_multicare_dataloader
from generate_embeddings import get_caption_embedding
from image_encoder import ImageEncoder
from caption_lstm import CaptionLSTM
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output


# Combined loss - Cosine + MSE
def hybrid_loss(output, target):
    mse = nn.MSELoss()(output, target)
    cosine = 1 - nn.CosineSimilarity()(output, target).mean()
    return 0.7*cosine + 0.3*mse  # Weighted combination

def train_model(num_epochs, loader, image_encoder, caption_lstm, optimizer, scheduler,
                tokenizer, bert_model, device, grad_clip_value=1.0):

    # Initialize tracking
    train_losses = []
    epoch_times = []
    best_loss = float('inf')

    # Create figure for live plotting
    plt.figure(figsize=(12, 5))

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        batch_losses = []

        # Initialize progress bar
        pbar = tqdm(enumerate(loader), total=len(loader),
                    desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')

        for batch_idx, batch in pbar:
            # Get data
            images = batch['image'].to(device)
            captions = batch['caption']

            # Forward pass
            image_features = image_encoder(images)
            caption_embeddings = torch.stack([
                get_caption_embedding(cap, tokenizer, bert_model, device)
                for cap in captions
            ])

            # Enhanced LSTM forward
            outputs = caption_lstm(
                caption_embeddings.unsqueeze(1),  # [batch, 1, 768]
                image_features
            )

            # Loss calculation
            loss = hybrid_loss(outputs, caption_embeddings)
            running_loss += loss.item()
            batch_losses.append(loss.item())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(caption_lstm.parameters(), grad_clip_value)
            torch.nn.utils.clip_grad_norm_(image_encoder.parameters(), grad_clip_value)

            optimizer.step()
            scheduler.step()

            # Update progress bar
            avg_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({
                'batch_loss': f'{loss.item():.4f}',
                'epoch_avg': f'{avg_loss:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

            # Live plot every 20% of epoch
            if (batch_idx + 1) % max(1, len(loader)//5) == 0:
                update_live_plot(epoch+1, batch_losses, epoch_times, len(loader))

        # Epoch complete
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        epoch_loss = running_loss / len(loader)
        train_losses.append(epoch_loss)

        # Update best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': caption_lstm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_model.pth')

        # Print epoch summary
        print(f'\nEpoch {epoch+1} completed:')
        print(f'Avg Loss: {epoch_loss:.4f} | Time: {epoch_time:.1f}s')
        print(f'Estimated remaining: {estimate_remaining_time(epoch, num_epochs, epoch_times)}')

        # Final plot update
        update_live_plot(epoch+1, batch_losses, epoch_times, len(loader), final_update=True)

    return train_losses

def update_live_plot(current_epoch, batch_losses, epoch_times, batches_per_epoch, final_update=False):
    clear_output(wait=True)
    plt.clf()

    # Create subplots
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    # Plot batch losses
    ax1.plot(batch_losses, label='Batch Loss')
    ax1.set_title(f'Epoch {current_epoch} Batch Losses')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)

    # Plot epoch trends if available
    if len(epoch_times) > 1:
        ax2.plot(range(1, current_epoch+1), epoch_times, 'bo-')
        ax2.set_title('Training Progress')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Time (s)')
        ax2.grid(True)

    plt.tight_layout()
    if final_update:
        plt.show()
    else:
        plt.pause(0.1)

def estimate_remaining_time(current_epoch, total_epochs, epoch_times):
    if len(epoch_times) < 2:
        return "Calculating..."

    avg_time = sum(epoch_times) / len(epoch_times)
    remaining = avg_time * (total_epochs - current_epoch - 1)

    if remaining > 3600:
        return f'{remaining/3600:.1f} hours'
    elif remaining > 60:
        return f'{remaining/60:.1f} minutes'
    else:
        return f'{remaining:.0f} seconds'

# Initialize everything
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Models
image_encoder = ImageEncoder(embed_size=256).to(device)
caption_lstm = CaptionLSTM(hidden_size=1024, num_layers=2).to(device)

# Partial unfreeze for image encoder
for param in image_encoder.resnet[-2:].parameters():
    param.requires_grad = True

# Optimizer with differential learning rates
optimizer = optim.Adam([
    {'params': caption_lstm.parameters(), 'lr': 3e-4},
    {'params': image_encoder.fc.parameters(), 'lr': 1e-4},
    {'params': image_encoder.resnet[-2:].parameters(), 'lr': 5e-5}
])

# Scheduler
warmup_epochs = 3
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return 0.01 + 0.99*(epoch/warmup_epochs)
    return 0.5*(1 + math.cos(math.pi*(epoch - warmup_epochs)/(num_epochs - warmup_epochs)))
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

caption_filters = [
        {'field': 'label', 'string_list': ['mri', 'head']},
        {'field': 'caption', 'string_list': ['showing', 'demonstrates', 'reveals'], 'operator': 'any'}
    ]

# Load data
loader = get_multicare_dataloader(
    dataset_name='med_test',
    batch_size=8,
    create_new=True,
    filters=caption_filters
)

# Start training
num_epochs = 1
tokenizer = BertTokenizer.from_pretrained("MediCareBertTokenizer")
bert_model = BertModel.from_pretrained("MediCareBertModel")

train_losses = train_model(
    num_epochs=num_epochs,
    loader=loader,
    image_encoder=image_encoder,
    caption_lstm=caption_lstm,
    optimizer=optimizer,
    scheduler=scheduler,
    tokenizer=tokenizer,
    bert_model=bert_model,
    device=device
)