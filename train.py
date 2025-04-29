import torch
torch.set_num_threads(2)
import numpy as np
from transformers import LongformerForMaskedLM, BigBirdForMaskedLM, BigBirdTokenizer, \
                                AutoModelForMaskedLM ,BigBirdConfig, AutoTokenizer, LongformerTokenizer, \
                                AutoImageProcessor, ViTForMaskedImageModeling, LongformerConfig
import argparse
from data_loader import EGMDataset, EGMIMGDataset, EGMTSDataset
from torch.utils.data import DataLoader
import gc
from torch.optim import Adam
import torch.nn as nn
import matplotlib.pyplot as plt
import os

from optim import ScheduledOptim, early_stopping
from models import VITModel, TimeSeriesModel
from runners import trainer, validate


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--lr', type = float, default = 1e-4, help='Please choose the learning rate')
    parser.add_argument('--patience', type = int, default = 5, help = 'Please choose the patience of the early stopper')
    parser.add_argument('--signal_size', type = int, default = 250, help = 'Please choose the signal size')
    parser.add_argument('--device', type = str, default = 'cuda', help = 'Please choose the type of device' )
    parser.add_argument('--warmup', type = int, default = 2000, help = 'Please choose the number of warmup steps for the optimizer' )
    parser.add_argument('--epochs', type = int, default = 100, help = 'Please choose the number of epochs' )
    parser.add_argument('--batch', type = int, default = 2, help = 'Please choose the batch size')
    parser.add_argument('--weight_decay', type = float, default = 1e-2, help = 'Please choose the weight decay')
    parser.add_argument('--model', type = str, default = 'big', help = 'Please choose which model to use')
    parser.add_argument('--use_ce', action='store_true', help = 'Please choose whether to use CE loss or not')  
    parser.add_argument('--mask', type=float, default=0.15, help = 'Pleasee choose percentage to mask for signal')
    parser.add_argument('--mlm_weight', type = float, default = 1.0, help = 'Please choose the weight for the mlm loss')
    parser.add_argument('--ce_weight', type = float, default = 1.0, help = 'Please choose the weight for the ce loss')
    parser.add_argument('--TS', action='store_true', help = 'Please choose whether to do Token Substitution')
    parser.add_argument('--TA', action='store_true', help = 'Please choose whether to do Token Addition')
    parser.add_argument('--LF', action='store_true', help = 'Please choose whether to do label flipping')
    parser.add_argument('--toy', action = 'store_true', help = 'Please choose whether to use a toy dataset or not')
    parser.add_argument('--norm_loss', type=float, default=0.1, help = "Specify norm loss coefficient")
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

def does_checkpoint_exist(directory_path):
    if os.path.exists(f'./{directory_path}/current_epoch.chkpt'):
        return True
    else:
        return False

def main():
    args = get_args()
    directory_path = f'./runs/checkpoint/saved_best_{args.lr}_{args.batch}_{args.patience}_{args.weight_decay}_{args.model}_{args.use_ce}_{args.mask}_{args.mlm_weight}_{args.ce_weight}_{args.toy}_{args.norm_loss}_{args.TS}_{args.TA}_{args.LF}'
    current_epoch_path = f'./runs/checkpoint/saved_current_epoch_{args.lr}_{args.batch}_{args.patience}_{args.weight_decay}_{args.model}_{args.use_ce}_{args.mask}_{args.mlm_weight}_{args.ce_weight}_{args.toy}_{args.norm_loss}_{args.TS}_{args.TA}_{args.LF}'
    ensure_directory_exists(directory_path)
    ensure_directory_exists(current_epoch_path)

    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(2)
    device = torch.device(args.device)
    print(device)
    print('Loading Data...')
    print(f'CE being used: {args.use_ce}')
    
    train = np.load('./data/train_intra.npy', allow_pickle = True).item()
    val = np.load('./data/val_intra.npy', allow_pickle = True).item()
    
    if args.toy:
        train = create_toy(train, [0, 1])
        val = create_toy(val, [14])

    print('Creating Custom Tokens...')
    
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
        tokenizer.add_tokens(custom_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = model.config.hidden_size
            
    if args.model == 'raw_big':
        configuration = BigBirdConfig(attention_type = 'original_full')
        model = BigBirdForMaskedLM(config = configuration).to(device)
        tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")
        tokenizer.add_tokens(custom_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = model.config.hidden_size
        
    if args.model =='clin_bird':
        model = AutoModelForMaskedLM.from_pretrained("yikuan8/Clinical-BigBird").to(device)
        model.config.attention_type = 'original_full'
        tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-BigBird")
        tokenizer.add_tokens(custom_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = model.config.hidden_size
        
    if args.model =='clin_long':
        model = AutoModelForMaskedLM.from_pretrained("yikuan8/Clinical-Longformer").to(device)
        tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
        tokenizer.add_tokens(custom_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = model.config.hidden_size
        
    if args.model == 'vit':
        tokenizer = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        pt_model = ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
        args.num_patches = (pt_model.config.image_size // pt_model.config.patch_size) ** 2
        model_hidden_size = pt_model.config.hidden_size
        model = VITModel(pt_model, model_hidden_size, 2).to(device)
        
    if args.model == 'big_ts':
        pt_model = BigBirdForMaskedLM.from_pretrained("google/bigbird-roberta-base").to(device)
        pt_model.config.attention_type = 'original_full'
        model_hidden_size = pt_model.config.hidden_size
        model = TimeSeriesModel(pt_model, model_hidden_size, 2).to(device)
        
    if args.model == 'long_ts':
        pt_model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096").to(device)
        model_hidden_size = pt_model.config.hidden_size
        model = TimeSeriesModel(pt_model, model_hidden_size, 2).to(device)
        
    if args.model == 'long':
        model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096").to(device)
        tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        tokenizer.add_tokens(custom_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = model.config.hidden_size
            
    if args.model == 'raw_long':
        configuration = LongformerConfig()
        model = LongformerForMaskedLM(config = configuration).to(device)
        tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        tokenizer.add_tokens(custom_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = model.config.hidden_size
    
    print('Creating Dataset and DataLoader...')
    if args.model == 'vit':
        train_dataset = EGMIMGDataset(train, tokenizer, args = args)        
        val_dataset = EGMIMGDataset(val, tokenizer, args = args)
    elif args.model == 'big_ts' or args.model == 'long_ts':
        train_dataset = EGMTSDataset(train, args = args)        
        val_dataset = EGMTSDataset(val, args = args)
    else:
        train_dataset = EGMDataset(train, tokenizer, args = args)        
        val_dataset = EGMDataset(val, tokenizer, args = args)
    
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle = True)   
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle = True)
    
    optimizer = ScheduledOptim(
    Adam(filter(lambda x: x.requires_grad, model.parameters()),
        betas=(0.9, 0.98), eps=1e-4, lr = args.lr, weight_decay=args.weight_decay), model_hidden_size, args.warmup)
    if args.use_ce:
        ce_loss = nn.CrossEntropyLoss(reduction = 'none')
    else:
        ce_loss = None

    train_losses = []
    val_losses = []
    all_epochs = []
    starting_epoch = -1
    if(does_checkpoint_exist(current_epoch_path)):
        checkpoint = torch.load(f'./{current_epoch_path}/current_epoch.chkpt',weights_only=False)
        model.load_state_dict(checkpoint['model'])
        starting_epoch = checkpoint['epoch']
        print(f"resuming from epoch {starting_epoch + 1}...")
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        all_epochs = checkpoint['all_epochs']
        optimizer._optimizer.load_state_dict(checkpoint['optimizer'])

        for key, value in checkpoint['scheduler'].items():
            if hasattr(optimizer, key):
                setattr(optimizer, key, value)
        print(all_epochs)
    else:
        print("starting from epoch 0...")
    for epoch in range(starting_epoch + 1, args.epochs):
        model_state_dict = model.state_dict()
            
        all_epochs.append(epoch)
        train_loss = trainer(model, train_loader, optimizer, device, args, ce_loss)
        print(f"Training - Epoch: {epoch+1},Train Loss: {train_loss}")
        train_losses.append(train_loss)
            
        val_loss = validate(model, val_loader, device, args, ce_loss)
        print(f"Evaluation - Epoch: {epoch+1}, Val Loss: {val_loss}")
        val_losses.append(val_loss)
            
        model_state_dict = model.state_dict()
                
        checkpoint = {
            'model' : model_state_dict,
            'config_file' : 'config',
            'epoch' : epoch
        }
            
        if val_loss <= min(val_losses):
            torch.save(checkpoint, f'./{directory_path}/best_checkpoint.chkpt')
            print('    - [Info] The best checkpoint file has been updated.')
        checkpoint = {
            'model' : model_state_dict,
            'config_file' : 'config',
            'optimizer': optimizer._optimizer.state_dict(),  # Save wrapped Adam optimizer
            'scheduler': optimizer.__dict__,
            'epoch' : epoch,
            'val_losses' : val_losses,
            'train_losses': train_losses,
            'all_epochs': all_epochs
        }
        torch.save(checkpoint, f'./{current_epoch_path}/current_epoch.chkpt')
        print("the latest epoch's checkpoint has been saved")
        early_stop = early_stopping(val_losses, patience = args.patience, delta = 0.01)
        
        if early_stop:
            print('Validation loss has stopped decreasing. Early stopping...')
            break   
    
    fig1 = plt.figure('Figure 1')
    plt.plot(train_losses, label = 'train')
    plt.plot(val_losses, label= 'valid')
    plt.xlabel('epoch')
    plt.ylim([0.0, max(train_losses)])
    plt.ylabel('loss')
    plt.legend(loc ="upper right")
    plt.title('loss change curve')
    plt.savefig(f'./{directory_path}/loss_plot.png')
    plt.close()

if __name__ == '__main__':
    print(torch.cuda.is_available())
    main()
