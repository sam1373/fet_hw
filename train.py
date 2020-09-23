import torch

from data import FetDataset
from model import LstmFet
from util import (load_word_embed,
                  get_label_vocab,
                  calculate_macro_fscore)

from torch.utils.tensorboard import SummaryWriter

from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers import get_linear_schedule_with_warmup

from sklearn.metrics import f1_score, accuracy_score

import os.path

def print_result(rst, vocab, mention_ids):
    rev_vocab = {i: s for s, i in vocab.items()}
    for sent_rst, mention_id in zip(rst, mention_ids):
        labels = [rev_vocab[i] for i, v in enumerate(sent_rst) if v == 1]
        print(mention_id, ', '.join(labels))


gpu = True

batch_size = 1000
# Because FET datasets are usually large (1m+ sentences), it is infeasible to 
# load the whole dataset into memory. We read the dataset in a streaming way.
buffer_size = 1000 * 2000

eval_steps = 500

#train_file = '/home/samuel/Downloads/hw2.data/en.train.json'
train_file = '/shared/nas/data/m1/yinglin8/projects/fet/data/aida_2020/hw2/en.train.ds.json'
#dev_file = '/home/samuel/Downloads/hw2.data/en.dev.json'
dev_file = '/shared/nas/data/m1/yinglin8/projects/fet/data/aida_2020/hw2/en.dev.ds.json'
#test_file = '/home/samuel/Downloads/hw2.data/en.test.json'
test_file = '/shared/nas/data/m1/yinglin8/projects/fet/data/aida_2020/hw2/en.test.ds.json'

embed_file = '/home/samuel/Downloads/enwiki.skip.size200.win10.neg15.sample1e-5.min15.txt'
#embed_file = '/home/samuel/Downloads/glove.840B.300d.txt'
# '/shared/nas/data/m1/yinglin8/embedding/enwiki.cbow.100d.case.txt'
embed_dim = 200
hidden_dim = 128
char_embed_dim = 64
embed_dropout = 0.3
lstm_dropout = 0.3

lr = 1e-3
weight_decay = 1e-3
max_epoch = 100

# Datasets
train_set = FetDataset(train_file)
dev_set = FetDataset(dev_file)
test_set = FetDataset(test_file)

# Load word embeddings from file
# If the word vocab is too large for your machine, you can remove words not
# appearing in the data set.
word_num = 833977
print('Loading word embeddings from %s' % embed_file)
word_embed, word_vocab = load_word_embed(embed_file,
                                         embed_dim,
                                         skip_first=True)

# Scan the whole dateset to get the label set. This step may take a long 
# time. You can save the label vocab to avoid scanning the dataset 
# repeatedly.
print('Collect fine-grained entity labels')
label_vocab = get_label_vocab(dev_file, test_file)
label_num = len(label_vocab)
vocabs = {'word': word_vocab, 'label': label_vocab}

# Build the model
print('Building the model')
model = LstmFet(word_num, label_num, embed_dim, hidden_dim, char_embed_dim, embed_dropout, lstm_dropout)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data.shape)

# Optimizer: Adam with decoupled weight decay
optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad,
                                     model.parameters()),
                              lr=lr,
                              weight_decay=weight_decay)

#schedule = get_linear_schedule_with_warmup(optimizer,
#                                           num_warmup_steps=5000,
#                                           num_training_steps=100000)

schedule = ReduceLROnPlateau(optimizer, 'max', patience=5, cooldown=3, verbose=True, threshold=0.001)

writer = SummaryWriter()

model_path = "model.pt"

if os.path.isfile(model_path):
    #model = torch.load(model_path)
    sd = torch.load(model_path, map_location='cuda:0')
    model.load_state_dict(sd)
else:
    model.word_embed = word_embed

if gpu:
    model.cuda()

global_step = 0

rev_word_vocab = {i: s for s, i in word_vocab.items()}
rev_label_vocab = {i: s for s, i in label_vocab.items()}

best_dev_score = best_test_score = 0.5
for epoch in range(max_epoch):
    # print('Epoch %d' % epoch)

    # Training set
    losses = []
    for idx, batch in enumerate(train_set.batches(vocabs,
                                                  batch_size,
                                                  buffer_size,
                                                  shuffle=True,
                                                  gpu=gpu)):
        print('\rStep %d' % global_step, end='')
        optimizer.zero_grad()

        # Unpack the batch
        (token_idxs, labels,
         mention_mask, context_mask,
         mention_ids, mentions, seq_lens, mention_chars, chars_len) = batch

        loss, scores = model(token_idxs,
                             mention_mask,
                             context_mask,
                             labels,
                             seq_lens,
                             mention_chars, chars_len)

        #print(loss)

        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('epoch', epoch, global_step)
        writer.add_scalar('lr', lr, global_step)
        writer.add_scalar('train_loss', loss, global_step)

        if idx % eval_steps == 0:
            print()
            # Dev set
            best_dev = False
            dev_results = {'gold': [], 'pred': [], 'id': []}
            for batch in dev_set.batches(vocabs,
                                         batch_size // 10,
                                         buffer_size,
                                         gpu=gpu,
                                         max_len=1000):
                # Unpack the batch
                (token_idxs, labels,
                 mention_mask, context_mask,
                 mention_ids, mentions, seq_lens, mention_chars, chars_len) = batch

                predictions = model.predict(token_idxs,
                                            mention_mask,
                                            context_mask,
                                            seq_lens,
                                            mention_chars, chars_len)

                dev_results['gold'].extend(labels.int().tolist())
                dev_results['pred'].extend(predictions.int().tolist())
                dev_results['id'].extend(mention_ids)


            # precision, recall, fscore = calculate_macro_fscore(dev_results['gold'],
            #                                                dev_results['pred'])

            f1micro = f1_score(dev_results['gold'], dev_results['pred'], average='micro')
            f1macro = f1_score(dev_results['gold'], dev_results['pred'], average='macro')
            acc = accuracy_score(dev_results['gold'], dev_results['pred'])

            print('Dev Micro-F1 {:.2f} Macro-F1 {:.2f} Accuracy {:.2f}'.format(
                f1micro, f1macro, acc))

            writer.add_scalar('dev_f1_micro', f1micro, global_step)
            writer.add_scalar('dev_f1_macro', f1macro, global_step)
            writer.add_scalar('dev_acc', acc, global_step)

            schedule.step(f1micro)

            if f1micro > best_dev_score:
                best_dev_score = f1micro
                best_dev = True


                print('Saving best dev f1micro model')
                torch.save(model.state_dict(), model_path)

                sd = torch.load(model_path, map_location='cuda:0')
                model.load_state_dict(sd)

            f = open("test_preds/gs_" + str(global_step), "w")

            # Test set
            test_results = {'gold': [], 'pred': [], 'id': []}
            for batch in test_set.batches(vocabs,
                                          batch_size // 10,
                                          buffer_size,
                                          gpu=gpu,
                                          max_len=1000):
                # Unpack the batch
                (token_idxs, labels,
                 mention_mask, context_mask,
                 mention_ids, mentions, seq_lens, mention_chars, chars_len) = batch

                predictions = model.predict(token_idxs,
                                            mention_mask,
                                            context_mask,
                                            seq_lens,
                                            mention_chars, chars_len)

                test_results['gold'].extend(labels.int().tolist())
                test_results['pred'].extend(predictions.int().tolist())
                test_results['id'].extend(mention_ids)

                j = 7

                #for j in range(0, len(token_idxs), 20):
                sent = " ".join([rev_word_vocab[i.item()] for i in token_idxs[j]])
                mention = " ".join([rev_word_vocab[token_idxs[j][i].item()] for i in torch.nonzero(mention_mask[j])])
                labels_correct = " ".join([rev_label_vocab[i.item()] for i in torch.nonzero(labels[j])])
                labels_pred = " ".join([rev_label_vocab[i.item()] for i in torch.nonzero(predictions[j])])
                m_c = "".join(chr(i) if i != 127 else "~" for i in mention_chars[j][:chars_len[j]])

                f.write(sent + "\n" + mention + "\n" + m_c + "\n" + labels_correct + "\n" + labels_pred + "\n\n")

                writer.add_text("sentence", sent, global_step)
                writer.add_text("mention", mention, global_step)
                writer.add_text("mention_chars", m_c, global_step)
                writer.add_text("labels_correct", labels_correct, global_step)
                writer.add_text("labels_pred", labels_pred, global_step)

            f.close()

            f1micro = f1_score(test_results['gold'], test_results['pred'], average='micro')
            f1macro = f1_score(test_results['gold'], test_results['pred'], average='macro')
            acc = accuracy_score(test_results['gold'], test_results['pred'])

            if best_dev:
                print("Test score for best dev:")

            print('Test Micro-F1 {:.2f} Macro-F1 {:.2f} Accuracy {:.2f}'.format(
                f1micro, f1macro, acc))

        global_step += 1

    print()
    print('Loss: {:.4f}'.format(sum(losses) / len(losses)))

# print('Best macro F-score (dev): {:2.f}'.format(best_dev_score))
# print('Best macro F-score (test): {:2.f}'.format(best_test_score))
