import time
import os
import numpy as np
import torch.optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils import data
import argparse
import json
#import torchvision.transforms as transforms
from data.LEVIR_CC.LEVIRCC import LEVIRCCDataset_video
from data.Dubai_CC.DubaiCC import DubaiCCDataset
from model.model_encoder import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer_video
from utils import *
from model.video_encoder import Video_encoder, Sty_fusion

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

def main(args):
    """
    Training and validation.
    """
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if os.path.exists(args.savepath)==False:
        os.makedirs(args.savepath)
    # best_bleu4 = 0.4  # BLEU-4 score right now
    best_bleu4 = 0.0  # Fresh training
    start_epoch = 0
    with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
        word_vocab = json.load(f)
    # Initialize / load checkpoint  2644246
    if args.checkpoint is None:      
        video_encoder=Video_encoder()
        sty_fusion=Sty_fusion()
        sty_fusion_optimizer = torch.optim.Adam(sty_fusion.parameters(), lr=args.encoder_lr, weight_decay=1e-2)
        parameters = []
        for name, param in video_encoder.named_parameters():
            if 'att_liner' in name:
                parameters.append({'params': param, 'lr': 1e-6})
        print("Trainable layers in Video_encoder:")
        for name, param in video_encoder.named_parameters():
            if param.requires_grad:
                print(name)
        video_encoder_optimizer = torch.optim.Adam(parameters, lr=args.encoder_lr, weight_decay=1e-2)

        decoder = DecoderTransformer_video(encoder_dim=args.encoder_dim, feature_dim=args.feature_dim, vocab_size=len(word_vocab), max_lengths=args.max_length, word_vocab=word_vocab, n_head=args.n_heads,
                                    n_layers= args.decoder_n_layers, dropout=args.dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                            lr=args.decoder_lr)
    else:
        video_encoder=Video_encoder()
        decoder = DecoderTransformer_video(encoder_dim=args.encoder_dim, feature_dim=args.feature_dim, vocab_size=len(word_vocab), max_lengths=args.max_length, word_vocab=word_vocab, n_head=args.n_heads,
                                    n_layers= args.decoder_n_layers, dropout=args.dropout)
        sty_fusion=Sty_fusion()
        sty_fusion_optimizer = torch.optim.Adam(sty_fusion.parameters(), lr=args.encoder_lr, weight_decay=1e-2)
        checkpoint = torch.load(args.checkpoint)
        video_encoder.load_state_dict(checkpoint['video_encoder_dict'])
        parameters = []
        for name, param in video_encoder.named_parameters():
            if 'att_liner' in name:
                parameters.append({'params': param, 'lr': 1e-6})
        print("Trainable layers in Video_encoder:")
        for name, param in video_encoder.named_parameters():
            if param.requires_grad:
                print(name)
        decoder.load_state_dict(checkpoint['decoder_dict'])

        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                            lr=args.decoder_lr)
        video_encoder_optimizer = torch.optim.Adam(parameters, lr=args.encoder_lr, weight_decay=1e-2)
    # Move to GPU, if available
    video_encoder.cuda()
    decoder = decoder.cuda()
    sty_fusion.cuda()
    # Loss function
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # Custom dataloaders
    if args.data_name == 'LEVIR_CC':
        train_loader = data.DataLoader(
            LEVIRCCDataset_video(args.data_folder, args.list_path, 'train', args.token_folder,
                                 args.vocab_file, args.max_length, args.allow_unk, if_mask=True, 
                                #  mask_mode=args.mode
                                 ),
            batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = data.DataLoader(
            LEVIRCCDataset_video(args.data_folder, args.list_path, 'val', args.token_folder,
                                 args.vocab_file, args.max_length, args.allow_unk, if_mask=True, 
                                #  mask_mode=args.mode
                                 ),
            batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    elif args.data_name == 'Dubai_CC':
        train_loader = data.DataLoader(
            DubaiCCDataset(args.data_folder, args.list_path, 'train', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
            batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = data.DataLoader(
            DubaiCCDataset(args.data_folder, args.list_path, 'val', args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
            batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    

    decoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=5, gamma=0.5)
    l_resizeA = torch.nn.Upsample(size = (256, 256), mode ='bilinear', align_corners = True)
    l_resizeB = torch.nn.Upsample(size = (256, 256), mode ='bilinear', align_corners = True)
    index_i = 0
    hist = np.zeros((args.num_epochs * len(train_loader), 3))
    # Epochs
    
    for epoch in range(start_epoch, args.num_epochs):        
        # Batches
        tqdm_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Training]", dynamic_ncols=True)
        for id, (video_tensor, _, _, token, token_len, _,mask) in enumerate(tqdm_bar):
            start_time = time.time()
            decoder.train()  
            video_encoder.train() 
            sty_fusion.train()
            decoder_optimizer.zero_grad()
            video_encoder_optimizer.zero_grad()
            sty_fusion_optimizer.zero_grad()

            video_tensor=video_tensor.cuda()
            mask=mask.cuda()
            vedie_emb,_=video_encoder(video_tensor)
            vedie_emb=sty_fusion(vedie_emb,mask)

            token = token.squeeze(1).cuda()
            token_len = token_len.cuda()

            scores, caps_sorted, decode_lengths, sort_ind = decoder(vedie_emb, token, token_len)
            
            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            loss = criterion(scores, targets)
            # Back prop.
            loss.backward()
            # Clip gradients
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_value_(decoder.parameters(), args.grad_clip)
                # torch.nn.utils.clip_grad_value_(encoder_trans.parameters(), args.grad_clip)

            # Update weights  
            decoder_optimizer.step()         
            video_encoder_optimizer.step()           
            sty_fusion_optimizer.step()


            # Keep track of metrics     
            hist[index_i,0] = time.time() - start_time #batch_time        
            hist[index_i,1] = loss.item() #train_loss
            hist[index_i,2] = accuracy(scores, targets, 5) #top5
            index_i += 1   

            # Update TQDM BAR description
            tqdm_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Top-5 Acc": f"{accuracy(scores, targets, 5):.2f}"
            })

            # # Print status
            # if index_i % args.print_freq == 0:
            #     print('Epoch: [{0}][{1}/{2}]\t'
            #         'Batch Time: {3:.3f}\t'
            #         'Loss: {4:.4f}\t'
            #         'Top-5 Accuracy: {5:.3f}'.format(epoch, index_i, args.num_epochs*len(train_loader),
            #                                 np.mean(hist[index_i-args.print_freq:index_i-1,0])*args.print_freq,
            #                                 np.mean(hist[index_i-args.print_freq:index_i-1,1]),
            #                                 np.mean(hist[index_i-args.print_freq:index_i-1,2])))

        # One epoch's validation
        decoder.eval()  # eval mode (no dropout or batchnorm)
        sty_fusion.eval()
        video_encoder.eval()
        val_start_time = time.time()
        references = list()  # references (true captions) for calculating BLEU-4 score
        hypotheses = list()  # hypotheses (predictions)
        
        with torch.no_grad():
            # Batches
            # TQDM Validation
            tqdm_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Validation]", dynamic_ncols=True)
            for ind, (video_tensor, token_all, token_all_len, _, _, _,mask) in enumerate(tqdm_val):
                video_tensor=video_tensor.cuda()
                mask=mask.cuda()
                vedie_emb,_=video_encoder(video_tensor)
                vedie_emb=sty_fusion(vedie_emb,mask)
                token_all = token_all.squeeze(0).cuda()
                # Forward prop.
                
                seq = decoder.sample(vedie_emb, k=1)
                img_token = token_all.tolist()
                img_tokens = list(map(lambda c: [w for w in c if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}],
                        img_token))  # remove <start> and pads
                references.append(img_tokens)

                pred_seq = [w for w in seq if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
                hypotheses.append(pred_seq)
                
                assert len(references) == len(hypotheses)

                if ind % args.print_freq == 0:
                    pred_caption = ""
                    ref_caption = ""
                    for i in pred_seq:
                        pred_caption += (list(word_vocab.keys())[i]) + " "
                    ref_caption = ""
                    for i in img_tokens:
                        for j in i:
                            ref_caption += (list(word_vocab.keys())[j]) + " "
                        ref_caption += ".    "
                tqdm_val.set_postfix({"Pred": "✔", "Ref": "✔"})


            val_time = time.time() - val_start_time
            # Calculate evaluation scores
            
            score_dict = get_eval_score(references, hypotheses)
            Bleu_1 = score_dict['Bleu_1']
            Bleu_2 = score_dict['Bleu_2']
            Bleu_3 = score_dict['Bleu_3']
            Bleu_4 = score_dict['Bleu_4']
            Meteor = score_dict['METEOR']
            Rouge = score_dict['ROUGE_L']
            Cider = score_dict['CIDEr']
            print(
                f"Validation:\n"
                f"Time: {val_time:.3f}\t"
                f"BLEU-1: {Bleu_1:.4f}\t"
                f"BLEU-2: {Bleu_2:.4f}\t"
                f"BLEU-3: {Bleu_3:.4f}\t"
                f"BLEU-4: {Bleu_4:.4f}\t"
                f"Rouge: {Rouge:.4f}\t"
                f"Meteor: {Meteor:.4f}\t"
                f"Cider: {Cider:.4f}\t"
            )
        
        #Adjust learning rate
        decoder_lr_scheduler.step()
   
        if  Bleu_4 > best_bleu4:
            best_bleu4 = max(Bleu_4, best_bleu4)
            #save_checkpoint                
            print('Save Model')  
            state = {'video_encoder_dict': video_encoder.state_dict(), 
                    'sty_fusion_dict': sty_fusion.state_dict(),   
                    'decoder_dict': decoder.state_dict(),
                    }                     
            # model_name = 'MV_CC_'+str(args.data_name)+'_batchsize_'+str(args.train_batchsize)+'_'+str(args.network)+'Bleu_4_'+str(round(10000*Bleu_4))+'.pth'
            
            model_name = 'MV_CC_'+str(args.data_name)+'_batchsize_'+str(args.train_batchsize)+'_'+str(args.network)+'.pth'
            
            torch.save(state, os.path.join(args.savepath, model_name))
            print(os.path.join(args.savepath, model_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Changes_to_Captions')

    # Data parameters
    parser.add_argument(
        '--data_folder', default='./LEVIR-MCI-dataset/images', help='folder with data files')
    parser.add_argument('--list_path', default='./data/LEVIR_CC/', help='path of the data lists')
    parser.add_argument('--token_folder', default='./data/LEVIR_CC/tokens/', help='folder with token files')
    parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    parser.add_argument('--max_length', type=int, default=41, help='path of the data lists')
    parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
    parser.add_argument('--data_name', default="LEVIR_CC",help='base name shared by data files.')

    #parser.add_argument('--data_folder', default='./Dubai_CC/DubaiCC500impair/datasetDubaiCCPublic/imgs_tiles/RGB/',help='folder with data files')
    #parser.add_argument('--list_path', default='./data/Dubai_CC/', help='path of the data lists')
    #parser.add_argument('--token_folder', default='./data/Dubai_CC/tokens/', help='folder with token files')
    #parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    #parser.add_argument('--max_length', type=int, default=27, help='path of the data lists')
    #parser.add_argument('--allow_unk', type=int, default=0, help='if unknown token is allowed')
    #parser.add_argument('--data_name', default="Dubai_CC",help='base name shared by data files.')

    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id in the training.')
    parser.add_argument('--checkpoint', default=None, help='path to checkpoint, None if none.')
    parser.add_argument('--print_freq',type=int, default=100, help='print training/validation stats every __ batches')
    # Training parameters
    parser.add_argument('--fine_tune_encoder', type=bool, default=True, help='whether fine-tune encoder or not')    
    parser.add_argument('--train_batchsize', type=int, default=32, help='batch_size for training')
    parser.add_argument('--network', default='resnet101', help='define the encoder to extract features')
    parser.add_argument('--encoder_dim',default=2048, help='the dimension of extracted features using different network')
    parser.add_argument('--feat_size', default=16, help='define the output size of encoder to extract features')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--workers', type=int, default=2, help='for data-loading; right now, only 0 works with h5pys in windows.')
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate for encoder if fine-tuning.')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='learning rate for decoder.')
    parser.add_argument('--grad_clip', type=float, default=None, help='clip gradients at an absolute value of.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    # Validation
    parser.add_argument('--val_batchsize', type=int, default=1, help='batch_size for validation')
    parser.add_argument('--savepath', default="./models_checkpoint/")
    # Model parameters
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--attention_dim', type=int, default=2048)
    parser.add_argument('--feature_dim', type=int, default=2048)
    args = parser.parse_args()
    print(f"Model save_path: {args.savepath}")
    main(args)
