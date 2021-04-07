
from argparse import ArgumentParser
from torch.nn import Transformer
import torchtext
import torch
import os
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torchtext.utils import download_from_url, extract_archive
import io
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from einops import rearrange
import math
import torch.distributed as dist
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import time
from torch.nn.parallel import DistributedDataParallel as DDP
from TransformerModel import TransformerTranslator
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

def data_process(filepaths, en_vocab, de_vocab,en_tokenizer,de_tokenizer):
    raw_de_itr = iter(io.open(filepaths[0],encoding="utf8"))
    raw_en_itr = iter(io.open(filepaths[1],encoding="utf8"))
    data =[]
    for(raw_de,raw_en) in zip(raw_de_itr,raw_en_itr):
        de_tensor_ = torch.tensor([de_vocab[token] for token in de_tokenizer(raw_de)],dtype=torch.long)
        en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(raw_en)],
                            dtype=torch.long)
        data.append((de_tensor_, en_tensor_))
    return data

def build_vocab(filepath,tokenizer):
        counter = Counter()
        with io.open(filepath,encoding="utf8") as f:
            for string_ in f:
                counter.update(tokenizer(string_))
        return Vocab(counter,specials=['<unk>','<pad>','<sos>','<eos>'])


def generate_batch(data_batch):

    de_batch, en_batch = [], []
    src_pad_masks,tgt_pad_masks = [],[]
    for (de_item, en_item) in data_batch:
        de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
        en_batch.append(en_item)
    de_batch = pad_sequence(de_batch, batch_first=True,padding_value=PAD_IDX)
    en_batch = pad_sequence(en_batch, batch_first=True,padding_value=PAD_IDX)
    for en_item in en_batch:
        curr_mask = en_item == PAD_IDX
        src_pad_masks.append(curr_mask)
    src_pad_masks = torch.stack(src_pad_masks)
    for de_item in de_batch:
        curr_mask = torch.logical_or(de_item == PAD_IDX,de_item == EOS_IDX)[:-1]
        tgt_pad_masks.append(curr_mask)
    tgt_pad_masks = torch.stack(tgt_pad_masks)
    return en_batch,de_batch,src_pad_masks,tgt_pad_masks

def prepare_data(rank,world_size):
    url_base = 'https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/'
    train_urls = ('train.de', 'train.en')
    val_urls = ('newstest2015.de', 'newstest2015.en')
    test_urls = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')

    url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
    train_urls = ('train.de.gz', 'train.en.gz')
    val_urls = ('val.de.gz', 'val.en.gz')
    test_urls = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')

    train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
    val_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]
    test_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]
    de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
    en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

    de_vocab = build_vocab(train_filepaths[0],de_tokenizer)
    en_vocab = build_vocab(train_filepaths[1],en_tokenizer)
    global PAD_IDX, BOS_IDX, EOS_IDX
    PAD_IDX = de_vocab['<pad>']
    BOS_IDX = de_vocab['<sos>']
    EOS_IDX = de_vocab['<eos>']
    train_data = data_process(train_filepaths,en_vocab,de_vocab,en_tokenizer,de_tokenizer)
    val_data = data_process(val_filepaths,en_vocab,de_vocab,en_tokenizer,de_tokenizer)
    test_data = data_process(test_filepaths,en_vocab,de_vocab,en_tokenizer,de_tokenizer)
    BATCH_SIZE = 128
    train_sampler = DistributedSampler(train_data,num_replicas=world_size,rank=rank)
    valid_sampler = DistributedSampler(val_data,num_replicas=world_size,rank=rank)
    test_sampler = DistributedSampler(test_data,num_replicas=world_size,rank=rank)

    train_iter = DataLoader(dataset=train_data, sampler=train_sampler,batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=generate_batch,num_workers=0)
    valid_iter = DataLoader(dataset=val_data, sampler=valid_sampler,batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=generate_batch,num_workers=0)
    test_iter = DataLoader(dataset=test_data, sampler=test_sampler,batch_size=BATCH_SIZE,
                        shuffle=False, collate_fn=generate_batch,num_workers=0)

    return train_iter, valid_iter, test_iter,len(en_vocab),len(de_vocab)

def gen_nopeek_mask(length):
    mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return mask  

def evaluate(model,optim,criterion,epoch,eval_data,writer,rank):
    model.eval()
    total_loss = 0
    start_time = time.time()
    f = open("model_eval_output"+ str(rank)+".txt" ,"a")
    with torch.no_grad():
        for i,(en_batch,de_batch,src_pad_masks,tgt_pad_masks) in enumerate(eval_data):
            en_batch = en_batch.to(rank)
            de_batch = de_batch.to(rank)
            src_pad_masks = src_pad_masks.to(rank)
            tgt_pad_masks = tgt_pad_masks.to(rank)
            optim.zero_grad()
            tgt_in = de_batch[:,:-1]
            tgt_out = de_batch[:,1:]

            tgt_mask = gen_nopeek_mask(tgt_in.shape[1]).to(rank)
            mem_pad_mask = src_pad_masks.clone()

            output = model(en_batch,src_pad_masks,tgt_in,tgt_mask,tgt_pad_masks,mem_pad_mask)
            loss = criterion(rearrange(output, 'b t v -> (b t) v'), rearrange(tgt_out, 'b o -> (b o)'))
            total_loss += loss.item()
        elapsed = time.time() - start_time

        curr_loss = total_loss / len(eval_data)
        print("Eval Epoch: {:d} Loss: {:.2f} | Batches/sec: {:.2f}".format(epoch,curr_loss,len(eval_data) / elapsed))
        f.write("Eval Epoch: {:d} Loss: {:.2f} | Batches/sec: {:.2f}\n".format(epoch,curr_loss,len(eval_data) / elapsed))
        if(not writer is None):
            writer.add_scalar('Evaluation Loss',curr_loss,epoch)
            writer.add_scalar('Evaluation Speed',len(eval_data)/elapsed,epoch)
    f.close()


def train(model,optim,criterion,rank,writer,epochs,train_data, eval_data):
    log_interval = 100
    total_loss = 0
    start_time = time.time()
    num_batch = len(train_data)
    epoch_time = time.time()
    f = open("model_train_output"+ str(rank)+".txt" ,"a")
    for epoch in range(epochs):
        print("Epoch: ",epoch)
        model.train()
        dist.barrier()
        for i,(en_batch,de_batch,src_pad_masks,tgt_pad_masks) in enumerate(train_data):
            en_batch = en_batch.to(rank)
            de_batch = de_batch.to(rank)
            src_pad_masks = src_pad_masks.to(rank)
            tgt_pad_masks = tgt_pad_masks.to(rank)
            optim.zero_grad()
            tgt_in = de_batch[:,:-1]
            tgt_out = de_batch[:,1:]

            tgt_mask = gen_nopeek_mask(tgt_in.shape[1]).to(rank)
            mem_pad_mask = src_pad_masks.clone()

            output = model(en_batch,src_pad_masks,tgt_in,tgt_mask,tgt_pad_masks,mem_pad_mask)
            loss = criterion(rearrange(output, 'b t v -> (b t) v'), rearrange(tgt_out, 'b o -> (b o)'))
            total_loss += loss.detach().item()
            if(i % log_interval == 0 and i != 0):
                elapsed = time.time() - start_time
                total_batches = epoch*len(train_data) + i
                curr_loss = total_loss / log_interval
                
                print("Training Loss: {:.2f} | Batches/sec: {:.2f} | Total batches: {:d}".format(curr_loss,log_interval / elapsed,total_batches))
                f.write("Training Loss: {:.2f} | Batches/sec: {:.2f} | Total batches: {:d}\n".format(curr_loss,log_interval / elapsed,total_batches))
                if(not writer is None):
                    writer.add_scalar('Training Loss',curr_loss,total_batches)
                    writer.add_scalar('Training Speed',log_interval/elapsed,total_batches)
                total_loss = 0
                start_time = time.time()
            loss.backward()
            optim.step()
        epoch_elapsed = time.time() - epoch_time
        print("Epoch time: " , epoch_elapsed, "seconds")
        f.write("Epoch time: {:.5f} seconds\n".format(epoch_elapsed))

        epoch_time = time.time()
        if(not writer is None):
            writer.add_scalar("Epoch Training Time:",epoch_elapsed,epoch)
        evaluate(model,optim,criterion,epoch,eval_data,writer,rank)
    f.close()

def distributed_train(rank, world_size):
    print("RANK: ",rank)
    print("WORLDSIZE:", world_size)
    print("Inside Distributed Train")
    setup(rank,world_size)
    print("Dist initted")
    torch.manual_seed(0)
    #Hyperparameters
    train_iter, valid_iter, test_iter,in_token,out_token = prepare_data(rank,world_size)
    
    emsize = 768
    nhid = 256
    nlayers = 6
    nhead = 6
    dropout = 0.2
    
    print("Initializing Model")
    model = TransformerTranslator(in_token,out_token,emsize,nhead,nhid,nlayers,dropout).to(rank)
    print("Model Loaded")
    optim = torch.optim.SGD(model.parameters(),lr=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(rank)
    model = DDP(model,device_ids=[rank])
    print("Model distributed")


    currTime = datetime.now().strftime("%d%m%Y%H%M%S")
    if(rank == 0):
        writer = SummaryWriter('runs/'+ currTime)
    else:
        writer = None

    train(model,optim,criterion,rank,writer,40,train_iter,valid_iter)

    cleanup()

def setup(rank,world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		rank=rank,world_size=world_size                                            
    )

def cleanup():
    dist.destroy_process_group()

def main():
    world_size = 2
    print("Spawning processes")
    mp.spawn(distributed_train, nprocs=world_size, args=(world_size,),join=True)         



    

if __name__ == '__main__':
    main()







