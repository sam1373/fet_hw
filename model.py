import torch
import torch.nn as nn
import torch.nn.utils.rnn as R
from typing import Tuple

from el_system import *
from el import simpleel

class LstmFet(nn.Module):
    def __init__(self,
                 word_num,
                 label_num,
                 embed_dim,
                 hidden_dim,
                 char_embed_dim,
                 word_embed_dropout: float = 0,
                 lstm_dropout: float = 0):
        super().__init__()

        el_candidates_file = EL_CANDIDATES_DATA_FILE
        print('init el with {} ...'.format(el_candidates_file), end=' ', flush=True)
        el_system = simpleel.SimpleEL.init_from_candidiate_gen_pkl(el_candidates_file)
        print('done', flush=True)

        self.gres = GlobalRes(TYPE_VOCAB, WIKI_FETEL_WORDVEC_FILE)

        """
        person_type_id = self.gres.type_id_dict.get('/person')
        l2_person_type_ids, person_loss_vec = None, None
        if person_type_id is not None:
            l2_person_type_ids = el.__get_l2_person_type_ids(self.gres.type_vocab)
            person_loss_vec = el.get_person_type_loss_vec(
                l2_person_type_ids, self.gres.n_types, 2.0, model.device)
        """

        self.el_entityvec = ELDirectEntityVec(
            self.gres.n_types, self.gres.type_id_dict, el_system, WID_TYPES_FILE)


        self.word_embed = nn.Embedding(word_num, embed_dim)
        self.lstm = nn.LSTM(embed_dim + 1, hidden_dim,
                                  batch_first=True,
                                  bidirectional=True)

        self.output_linear = nn.Sequential(nn.Linear(hidden_dim * 5 + 128 + 1, hidden_dim), nn.LeakyReLU(),
                                           nn.Linear(hidden_dim, label_num))

        self.word_embed_dropout = nn.Dropout(word_embed_dropout)
        self.lstm_dropout = nn.Dropout(lstm_dropout)

        self.criterion = nn.MultiLabelSoftMarginLoss()

        self.char_embed = nn.Embedding(128, char_embed_dim, padding_idx=0)
        self.char_lstm = nn.LSTM(char_embed_dim, hidden_dim, batch_first=True)

    def get_el(self, mention_strs):

        entity_vecs, el_sgns, el_probs = get_entity_vecs_for_samples(
            self.el_entityvec, mention_strs, True, None, None, self.gres.type_vocab)

        return entity_vecs, el_sgns, el_probs

    def forward_nn(self,
                   inputs: torch.Tensor,
                   mention_mask: torch.Tensor,
                   context_mask: torch.Tensor,
                   seq_lens: torch.Tensor,
                   mention_chars: torch.Tensor,
                   chars_len) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Word index tensor for the input batch.
            mention_mask (torch.Tensor): A mention mask with the same size of
              `inputs`.
            context_mask (torch.Tensor): A context mask with the same size of
              `inputs`.
            seq_lens (torch.Tensor): A vector of sequence lengths.

            If a sequence has 6 tokens, where the 2nd token is a mention, and
            the longest sequence in the current batch has 8 tokens, the mention
            mask and context mask of this sequence are:
            - mention mask: [0, 1, 0, 0, 0, 0, 0, 0]
            - context mask: [1, 1, 1, 1, 1, 1, 0, 0]

        Returns:
            torch.Tensor: label scores. A NxM matrix where N is the batch size
              and M is the number of labels.
        """

        mention_strs = ["".join(map(chr, mention_chars[i][:chars_len[i].item() - 1])) for i in range(mention_chars.shape[0])]

        #print(mention_strs[:10])

        ev, _, el_prob = self.get_el(mention_strs)

        ev = torch.Tensor(ev).cuda()
        el_prob = torch.Tensor(el_prob).cuda()

        #print(ev[:10])
        #print(el_prob[:10])

        #print(ev.shape, el_prob.shape)


        chars_embed = self.char_embed(mention_chars)
        chars_packed = R.pack_padded_sequence(chars_embed, chars_len, batch_first=True, enforce_sorted=False)
        char_out, (c_h, _) = self.char_lstm(chars_packed)

        c_h = c_h.squeeze(0)

        #scores = self.output_linear(c_h)

        #return scores

        ###end

        inputs_embed = self.word_embed(inputs)
        inputs_embed = self.word_embed_dropout(inputs_embed)

        inputs_embed_marked = torch.cat((inputs_embed, mention_mask.unsqueeze(-1)), dim=2)

        lstm_in = R.pack_padded_sequence(inputs_embed_marked,
                                         seq_lens,
                                         batch_first=True)

        #print(context_mask[0])
        #print(mention_mask[0])
        #print(seq_lens[:10])

        #print(inputs_embed_marked.shape)
        #print(lstm_in.data.shape)
        #print(lstm_in.batch_sizes)

        lstm_out, (h, c) = self.lstm(lstm_in)
        lstm_out, _ = R.pad_packed_sequence(lstm_out, batch_first=True)

        lstm_out = self.lstm_dropout(lstm_out)

        # Average mention embedding
        mention_mask = ((1 - mention_mask) * -1e14).softmax(-1).unsqueeze(-1)
        mention_repr = (lstm_out * mention_mask).mean(1)

        # Average context embedding
        #context_mask = ((1 - context_mask) * -1e14).softmax(-1).unsqueeze(-1)
        #context_repr = (lstm_out * context_mask).sum(1)


        h = h.transpose(0, 1)

        context_repr = h.reshape(h.shape[0], -1)

        #print(context_repr.shape)

        #mention_original = (inputs_embed * mention_mask).sum(1)


        # Concatenate mention and context representations
        combine_repr = torch.cat([mention_repr, context_repr, c_h, ev, el_prob.unsqueeze(1)], dim=1)


        # Linear classifier
        scores = self.output_linear(combine_repr)

        return scores

    def forward(self,
                inputs: torch.Tensor,
                mention_mask: torch.Tensor,
                context_mask: torch.Tensor,
                labels: torch.Tensor,
                seq_lens: torch.Tensor,
                mention_chars: torch.Tensor,
                chars_len) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs (torch.Tensor): Word index tensor for the input batch.
            mention_mask (torch.Tensor): A mention mask with the same size of
            `inputs`.
            context_mask (torch.Tensor): A context mask with the same size of
            `inputs`.
            labels (torch.Tensor): A tensor of label vectors. The label vector
              for each mention is a binary vector where each value indicates
              whether the corresponding label is assigned to the mention or not.
            seq_lens (torch.Tensor): A vector of sequence lengths.
            mention_chars (torch.Tensor): Characters in mention

            If a sequence has 6 tokens, where the 2nd token is a mention, and
            the longest sequence in the current batch has 8 tokens, the mention
            mask and context mask of this sequence are:
            - mention mask: [0, 1, 0, 0, 0, 0, 0, 0]
            - context mask: [1, 1, 1, 1, 1, 1, 0, 0]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The first element is the loss.
              The second element are label scores, a NxM matrix where N is
              the batch size and M is the number of labels.
        """
        scores = self.forward_nn(inputs, mention_mask, context_mask, seq_lens, mention_chars, chars_len)
        loss = self.criterion(scores, labels)
        return loss, scores

    def predict(self,
                inputs: torch.Tensor,
                mention_mask: torch.Tensor,
                context_mask: torch.Tensor,
                seq_lens: torch.Tensor,
                mention_chars: torch.Tensor, chars_len,
                predict_top: bool = True) -> torch.Tensor:
        """Predict fine-grained entity types of a batch of mentions.

        Args:
            inputs (torch.Tensor): word index tensor for the input batch.
            mention_mask (torch.Tensor): a mention mask with the same size of
              `inputs`.
            context_mask (torch.Tensor): a context mask with the same size of
              `inputs`.
            seq_lens (torch.Tensor): A vector of sequence lengths.
            predict_top (bool, optional): if True, a label will be predicted
              even if its score (after sigmoid) is smaller than 0.5. Defaults
              to True.

            If a sequence has 6 tokens, where the 2nd token is a mention, and
            the longest sequence in the current batch has 8 tokens, the mention
            mask and context mask of this sequence are:
            - mention mask: [0, 1, 0, 0, 0, 0, 0, 0]
            - context mask: [1, 1, 1, 1, 1, 1, 0, 0]

        Returns:
            torch.Tensor: prediction result. A NxM matrix where N is the batch
              size and M is the number of labels. Label j is predicted for the
              i-th mention if the i,j element is 1.
        """
        self.eval()
        scores = self.forward_nn(inputs, mention_mask, context_mask, seq_lens, mention_chars,chars_len)

        predictions = (scores.sigmoid() > .5).int()

        if predict_top:
            _, highest = scores.max(dim=1)
            highest = highest.int().tolist()
            for i, h in enumerate(highest):
                predictions[i][h] = 1

        self.train()

        return predictions