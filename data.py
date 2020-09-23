import json
import torch
import random
from typing import Dict, Tuple, List


def chunk_list(lst: list, chunk_size: int):
    chunks = [lst[i: i + chunk_size] for i in range(0, len(lst), chunk_size)]
    return chunks


class FetDataset(object):
    def __init__(self,
                 path: str):
        """
        Args:
            path (str): Path to the data file.
        """
        self.path = path
    
    def batches(self,
                vocabs: Dict[str, Dict[str, int]],
                batch_size: int = 1000,
                buffer_size: int = 100000,
                shuffle: bool = False,
                gpu: bool = False,
                max_len: int = 128) -> Tuple:
        """Generate batches.

        Args:
            vocabs (Dict[str, Dict[str, int]]): A dict of word and label vocabs.
            batch_size (int, optional): Batch size. Defaults to 1000.
            buffer_size (int, optional): Buffer size. Defaults to 100000.
            shuffle (bool, optional): If True, instances will be shuffled within 
              each buffer. Defaults to False.
            gpu (bool, optional): Use GPU. Defaults to False.

        Yields:
            Tuple: A processed batch. See `process_batch()`.
        """
        buffer = []
        with open(self.path) as r:
            for line in r:
                # Parse each line
                inst = json.loads(line)
                # I simply ignore overlength sentences in the training set.
                # For dev and test sets I raise the max_len value and decrease
                # the batch size to avoid the OOM error.
                if len(inst['tokens']) > max_len:
                    continue
                # Append to the buffer
                buffer.append(inst)
                
                if len(buffer) == buffer_size:
                    # Shuffle the buffer
                    if shuffle:
                        random.shuffle(buffer)
                    # Generate batches
                    for batch in chunk_list(buffer, batch_size):
                        yield self.process_batch(batch, vocabs, gpu)
                    # Empty the buffer
                    buffer = []
        
        # The last buffer may not be full
        if buffer:
            if shuffle:
                random.shuffle(buffer)
            for batch in chunk_list(buffer, batch_size):
                yield self.process_batch(batch, vocabs, gpu)
       
    @staticmethod         
    def process_batch(batch: list,
                      vocabs: Dict[str, Dict[str, int]],
                      gpu: bool = True
                      ) -> (torch.Tensor, torch.Tensor, torch.Tensor,
                            torch.Tensor, List[str], List[str],
                            torch.Tensor):
        """Process a batch of instances.

        Args:
            batch (list): A list of instance.
            vocabs (Dict[str, Dict[str, int]]): a dict of word and label vocabs.
            gpu (bool, optional): Use GPU. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                  torch.Tensor, List[str], List[str]]: A tuple of the following
                  elements:
                  - token indices: Word indices for embedding lookup.
                  - labels: Mention labels. The label vector for each mention is
                    a binary vector where each value indicates whether the
                    corresponding label is assigned to the mention or not.
                  - mention mask: Mention mask tensor.
                  - context mask: Context mask tensor.
                  - mention ids: A list of mention ids.
                  - mentions: A list of mention texts.
                  - sequence lengths: A tensor of sequence lengths.
        """
        batch.sort(key=lambda x: len(x['tokens']), reverse=True)
        
        word_vocab = vocabs['word']
        label_vocab = vocabs['label']
        label_num = len(label_vocab)
        
        batch_token_idxs = []
        batch_mention_mask = []
        batch_context_mask = []
        batch_labels = []
        batch_mention_ids = []
        batch_mentions = []
        batch_seq_lens = []
        batch_mention_chars = []
        batch_mention_char_lens = []

        max_token_num = max(len(x['tokens']) for x in batch)
        max_char_num = 32
        for instance in batch:
            tokens = instance['tokens']
            token_num = len(tokens)
            token_idxs = [word_vocab.get(t, word_vocab.get(t.lower(), 0))
                          for t in tokens]
            # Pad token indices with 0's
            token_idxs = token_idxs + [0] * (max_token_num - token_num)
            
            for annotation in instance['annotations']:
                # If a sentence contains multiple mentions, token_idxs will be
                # appended multiple times. If you use an encoder with a large
                # number of parameters (e.g., bidirection LSTM, Bert, ELMo) to 
                # encode the sentence, a more efficient way is to run the encode
                # for each sentence once and duplicate the output.
                batch_token_idxs.append(token_idxs)
            
                # Generate mention mask and context mask
                start, end = annotation['start'], annotation['end']
                mention_mask = [1 if start <= i < end else 0
                                for i in range(max_token_num)]
                context_mask = [1] * token_num + [0] * (max_token_num - token_num)
                batch_mention_mask.append(mention_mask)
                batch_context_mask.append(context_mask)
                
                # Mention labels
                labels = [0] * label_num
                for label in annotation['labels']:
                    if label not in label_vocab:
                        labels[0] = 1
                    else:
                        labels[label_vocab.get(label)] = 1
                batch_labels.append(labels)
                
                batch_mention_ids.append(annotation['mention_id'])
                batch_mentions.append(annotation['mention'])
                batch_seq_lens.append(token_num)

                chars = []

                for i in annotation['mention']:
                    chars.append(min(127, ord(i)))

                chars.append(ord("."))
                chars = chars[:max_char_num]
                chars_len = len(chars)
                chars = chars + [0] * (max_char_num - chars_len)

                batch_mention_chars.append(chars)
                batch_mention_char_lens.append(chars_len)
                
        # Convert to tensors
        batch_token_idxs = torch.LongTensor(batch_token_idxs)
        batch_labels = torch.FloatTensor(batch_labels)
        batch_mention_mask = torch.FloatTensor(batch_mention_mask)
        batch_context_mask = torch.FloatTensor(batch_context_mask)
        batch_seq_lens = torch.LongTensor(batch_seq_lens)
        batch_mention_chars = torch.LongTensor(batch_mention_chars)
        batch_mention_char_lens = torch.LongTensor(batch_mention_char_lens)
        if gpu:
            # Move to GPU
            batch_token_idxs = batch_token_idxs.cuda()
            batch_labels = batch_labels.cuda()
            batch_mention_mask = batch_mention_mask.cuda()
            batch_context_mask = batch_context_mask.cuda()
            batch_seq_lens = batch_seq_lens.cuda()
            batch_mention_chars = batch_mention_chars.cuda()
            batch_mention_char_lens = batch_mention_char_lens.cuda()

        return (batch_token_idxs, batch_labels,
                batch_mention_mask, batch_context_mask,
                batch_mention_ids, batch_mentions,
                batch_seq_lens, batch_mention_chars, batch_mention_char_lens)

