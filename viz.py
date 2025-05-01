import torch
torch.set_num_threads(2)
import numpy as np
from transformers import BigBirdForMaskedLM, LongformerForMaskedLM, BigBirdTokenizer, BigBirdForQuestionAnswering, \
                        AutoModelForMaskedLM, LongformerTokenizer, AutoTokenizer, BigBirdConfig, AutoImageProcessor, \
                            ViTForMaskedImageModeling, LongformerConfig
import argparse
from data_loader import EGMDataset, EGMIMGDataset, EGMTSDataset
from models import VITModel, TimeSeriesModel
from torch.utils.data import DataLoader
import gc
from runners import inference
import os
import matplotlib_inline
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import random


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--lr', type = float, default = 1e-4, help='Please choose the learning rate')
    parser.add_argument('--patience', type = int, default = 5, help = 'Please choose the patience of the early stopper')
    parser.add_argument('--signal_size', type = int, default = 250, help = 'Please choose the signal size')
    parser.add_argument('--device', type = str, default = 'cuda:1', help = 'Please choose the type of device' )
    parser.add_argument('--warmup', type = int, default = 2000, help = 'Please choose the number of warmup steps for the optimizer' )
    parser.add_argument('--epochs', type = int, default = 50, help = 'Please choose the number of epochs' )
    parser.add_argument('--batch', type = int, default = 2, help = 'Please choose the batch size')
    parser.add_argument('--weight_decay', type = float, default = 1e-2, help = 'Please choose the weight decay')
    parser.add_argument('--checkpoint', type = str, default = None, help = 'Please choose the path to the checkpoint to infer on')
    parser.add_argument('--model', type = str, default = 'big', help = 'Please choose which model to use')
    parser.add_argument('--mask', type=float, default=0.15, help = 'Pleasee choose percentage to mask for signal')
    parser.add_argument('--TS', action='store_true', help = 'Please choose whether to do Token Substitution')
    parser.add_argument('--TA', action='store_true', help = 'Please choose whether to do Token Addition')
    parser.add_argument('--LF', action='store_true', help = 'Please choose whether to do label flipping')    
    parser.add_argument('--toy', action = 'store_true', help = 'Please choose whether to use a toy dataset or not')
    parser.add_argument('--inference', action='store_true', help = 'Please choose whether it is inference or not')
    return parser.parse_args()

def create_toy(dataset, spec_ind):
    toy_dataset = {}
    for i in dataset.keys():
        _, placement, _, _ = i
        if placement in spec_ind:
            toy_dataset[i] = dataset[i]    
    return toy_dataset

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")

if __name__ == '__main__':

    gc.collect()
    torch.cuda.empty_cache()
    args = get_args()
    torch.manual_seed(2)
    device = torch.device(args.device)
    print(device)
    print('Loading Data...')

    test = np.load('./data/test_intra.npy', allow_pickle = True).item()
    
    if args.toy:
        test = create_toy(test, [18])

    print('Creating Custom Tokens')
    custom_tokens = [
        f"signal_{i}" for i in range(args.signal_size+1)
    ] + [
        f"afib_{i}" for i in range(2)
    ]
    if args.TA:
        custom_tokens += [
        f"augsig_{i}" for i in range(args.signal_size+1)
    ]

    print('Initalizing Model...')
    if args.model == 'big':
        model = BigBirdForMaskedLM.from_pretrained("google/bigbird-roberta-base").to(device)
        model.config.attention_type = 'original_full'
        tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")
        num_added_tokens = tokenizer.add_tokens(custom_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = model.config.hidden_size   
   
    if args.model == 'long':
        model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096",output_attentions=True).to(device)
        tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        num_added_tokens = tokenizer.add_tokens(custom_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = model.config.hidden_size
       
    if args.model =='clin_bird':
        model = AutoModelForMaskedLM.from_pretrained("yikuan8/Clinical-BigBird").to(device)
        model.config.attention_type = 'original_full'
        tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-BigBird")
        num_added_tokens = tokenizer.add_tokens(custom_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = model.config.hidden_size
        
    if args.model =='clin_long':
        model = AutoModelForMaskedLM.from_pretrained("yikuan8/Clinical-Longformer").to(device)
        tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
        num_added_tokens = tokenizer.add_tokens(custom_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = model.config.hidden_size
        
    print('Creating Dataset and DataLoader...')
    
    test_data = np.load('./data/test_intra.npy', allow_pickle=True).item()
    test_dataset = EGMDataset(test_data, tokenizer, args=args)
    
    checkpoint = torch.load(f'./runs/checkpoint/{args.checkpoint}/best_checkpoint.chkpt', map_location = args.device)
    model.load_state_dict(checkpoint['model'])
    sample = test_dataset[0]
    sample = test_dataset[0]
    input_ids = sample[0].unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    # === Ensure length is multiple of 2 * attention_window ===
    attention_window = model.config.attention_window[0]
    required_multiple = 2 * attention_window
    seq_len = input_ids.shape[1]

    if seq_len % required_multiple != 0:
        pad_len = required_multiple - (seq_len % required_multiple)
        print(f"Padding input from {seq_len} to {seq_len + pad_len} for Longformer compatibility")
        input_ids = F.pad(input_ids, (0, pad_len), value=tokenizer.pad_token_id)
        attention_mask = F.pad(attention_mask, (0, pad_len), value=0)

    # === Run Inference with Attention for Figure 3 ===
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        attentions = outputs.attentions  # List of [batch, heads, seq, seq]
        attribution_scores = outputs.logits  # Assuming logits are used for attribution

    # Decode tokens back to strings
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # === Average Attention Scores for Figure 3 ===
    # We need to average across all heads (dim=1) and layers (dim=2)
    avg_attention = torch.stack(attentions).mean(dim=0)  # Shape: [1, heads, seq_len, seq_len]
    avg_attention = avg_attention.mean(dim=1).squeeze()  # Average over heads, resulting in shape [1, seq_len, seq_len]
    avg_attention = avg_attention.cpu().numpy()  # Convert to numpy

    # === Create Figure 3: Attention Scores Visualization ===
    plt.figure(figsize=(14, 10))
    signal_length = np.arange(len(avg_attention))

    # Plot the signal (amplitude)
    plt.plot(signal_length, input_ids[0].cpu().numpy(), label="EGM Signal Amplitude", color='blue', alpha=0.7)

    # Plot attention scores as a shaded area
    # We expand avg_attention to (seq_len, seq_len) so that it can be used with imshow
    plt.imshow(avg_attention, cmap="viridis", aspect="auto", extent=[0, len(avg_attention), -1, 1], alpha=0.5)

    plt.title("Figure 3: Attention Scores Over EGM Signal")
    plt.xlabel("Signal Length (Time Steps)")
    plt.ylabel("Signal Amplitude / Attention Scores")
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(f"New_attention_scores_egm_signal.png")
    plt.show()


    # === Randomly Mask Tokens for Figure 4 ===
    input_ids_masked = input_ids.clone()
    seq_len = input_ids.shape[1]
    mask_indices = random.sample(range(1, seq_len - 1), k=int(0.15 * seq_len))

    # Mask tokens
    for idx in mask_indices:
        input_ids_masked[0, idx] = tokenizer.mask_token_id  # Masking tokens (0-index batch)

    # === Run Model with Masked Tokens to Get Attribution Scores ===
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids_masked, attention_mask=attention_mask)
        attribution_scores = outputs.logits  # These could be softmax logits or raw scores

    # Since attribution_scores have shape (1, seq_len, vocab_size), we need to average or collapse
    # Let's take the maximum value along the vocab size axis (axis=2)
    avg_attribution = torch.max(attribution_scores, dim=2)[0]  # Shape: [1, seq_len]
    avg_attribution = avg_attribution.squeeze().cpu().numpy()  # Convert to numpy

    # === Create Figure 4: Attribution Scores Visualization ===
    plt.figure(figsize=(14, 10))
    signal_length = np.arange(len(avg_attribution))

    # Plot the EGM Signal
    plt.plot(signal_length, input_ids[0].cpu().numpy(), label="EGM Signal Amplitude", color='blue', alpha=0.7)

    # Plot attribution scores as a shaded area
    plt.fill_between(signal_length, 0, avg_attribution, color="orange", alpha=0.5, label="Attribution Scores")

    plt.title("Figure 4: Attribution Scores Over EGM Signal (Masked Tokens)")
    plt.xlabel("Signal Length (Time Steps)")
    plt.ylabel("Signal Amplitude / Attribution Scores")
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(f"New_attribution_scores_egm_signal.png")
    plt.show()