import time
import numpy as np
from torch import nn
import time
import os
import torch
import logging
from torch.optim import AdamW
from transformers import Trainer, TrainingArguments, BertTokenizer, BertModel, BertPreTrainedModel, BertConfig, \
    get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from transformers.utils.notebook import format_time
from modeling import BertForSeq
from data_process import InputDataSet,read_data,fill_paddings
from sklearn.metrics import confusion_matrix as cm
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(batch_size,EPOCHS):

    model = BertForSeq.from_pretrained('bert-base-uncased')

    train = read_data('./train_text_processed.csv')
    val = read_data('./test_text_processed.csv')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = InputDataSet(train, tokenizer, 512)
    val_dataset = InputDataSet(val, tokenizer, 512)


    train_dataloader = DataLoader(train_dataset,batch_size)
    val_dataloader = DataLoader(val_dataset,batch_size)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    total_t0 = time.time()

    log = log_creater(output_dir='./cache/logs/')

    log.info("   Train batch size = {}".format(batch_size))
    log.info("   Total steps = {}".format(total_steps))
    log.info("   Training Start!")
    best_acc=0.0

    for epoch in range(EPOCHS):
        total_train_loss = 0
        t0 = time.time()
        model.to(device)
        model.train()
        for step, batch in enumerate(train_dataloader):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            model.zero_grad()

            predicts = model(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,labels=labels)
            loss_function=nn.CrossEntropyLoss()
            loss=loss_function(predicts, labels)

            total_train_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_time = format_time(time.time() - t0)

        log.info('====Epoch:[{}/{}] avg_train_loss={:.5f}===='.format(epoch+1,EPOCHS,avg_train_loss))
        log.info('====Training epoch took: {:}===='.format(train_time))
        log.info('Running Validation...')

        model.eval()
        avg_val_loss, avg_val_acc= evaluate(model, val_dataloader)
        val_time = format_time(time.time() - t0)
        log.info('====Epoch:[{}/{}] avg_val_loss={:.5f} avg_val_acc={:.5f}===='.format(epoch+1,EPOCHS,avg_val_loss,avg_val_acc))
        log.info('====Validation epoch took: {:}===='.format(val_time))
        log.info('')
        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save(model,'./cache/final_model_processed30.pth')
            print('Model Saved!')

    log.info('')
    log.info('   Training Completed!')
    print('Total training took{:} (h:mm:ss)'.format(format_time(time.time() - total_t0)))

def make_cm(matrix, col):

    confusion_matrix = pd.DataFrame(matrix,columns=col,index=col)
    return confusion_matrix

def evaluate(model,val_dataloader):
    total_val_loss = 0
    corrects = []
    for batch in val_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,labels=labels)


        logits=torch.argmax(outputs,dim=1)

        preds = logits.detach().cpu()
        labels_ids = labels.to('cpu')
        preds_numpy=preds.numpy()
        labels_ids_numpy=labels_ids.numpy()
        corrects.append((preds_numpy == labels_ids_numpy).mean())

        loss_function = nn.CrossEntropyLoss()
        loss=loss_function(outputs, logits)

        total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)
    avg_val_acc = np.mean(corrects)

    return avg_val_loss, avg_val_acc

def log_creater(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(output_dir, log_name)
    # creat a log
    log = logging.getLogger('train_log')
    log.setLevel(logging.DEBUG)

    # FileHandler
    file = logging.FileHandler(final_log_file)
    file.setLevel(logging.DEBUG)

    # StreamHandler
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s][line: %(lineno)d] ==> %(message)s')

    # setFormatter
    file.setFormatter(formatter)
    stream.setFormatter(formatter)

    # addHandler
    log.addHandler(file)
    log.addHandler(stream)

    log.info('creating {}'.format(final_log_file))
    return log


if __name__ == '__main__':
    train(batch_size=8,EPOCHS=30)
