import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionNetwork(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GraphAttentionNetwork, self).__init__()
        self.dropout = dropout

        self.attention_heads = nn.ModuleList([
            GraphAttentionHead(nfeat, nhid, dropout=dropout, alpha=alpha, concat=False)
            for _ in range(nheads)
        ])

    def forward(self, x, adj):
        # x = (N, l, in_features), adj = (N, l, l)
        y = torch.cat([att(x, adj) for att in self.attention_heads], dim=2)
        y = F.dropout(y, self.dropout, training=self.training)
        return y


class GraphAttentionHead(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionHead, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.a1 = nn.Parameter(torch.empty(size=(out_features, 1)))
        self.a2 = nn.Parameter(torch.empty(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h.shape: (bsz, N, in_features), Wh.shape: (bsz, N, out_features)
        Wh = torch.matmul(h, self.W)
        # a_input.shape : (bsz, N, N, 2 * out_features)
        # a_input = self._prepare_attentional_mechanism_input(Wh)

        # e.shape : (bsz, N, N)
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        f_1 = torch.matmul(Wh, self.a1)  # (bsz, N, 1)
        f_2 = torch.matmul(Wh, self.a2)  # (bsz, N, 1)
        e = self.leakyrelu(f_1 + f_2.transpose(1, 2))  # (bsz, N, N)

        zero_vec = -9e15 * torch.ones_like(e)
        # attention.shape : (bsz, N, N)
        attention = torch.where(adj > 0, e, zero_vec)
        # attention.shape : (bsz, N, N)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        # h_prime.shape: (bsz, N, in_features)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        bsz = Wh.size(0)
        N = Wh.size(1)  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (bsz, N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
        # all_combinations_matrix.shape == (bsz, N * N, 2 * out_features)

        return all_combinations_matrix.view(bsz, N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GATModel(nn.Module):

    def __init__(self, config):
        super(GATModel, self).__init__()

        head_dim = config.hidden_size // config.num_attention_heads
        self.max_syntactic_distance = config.max_syntactic_distance

        self.dep_tag_embed = None
        if config.use_dependency_tag:
            self.dep_tag_embed = nn.Embedding(
                config.dep_tag_vocab_size,
                config.hidden_size,
                padding_idx=0
            )
        self.pos_tag_embed = None
        if config.use_pos_tag:
            self.pos_tag_embed = nn.Embedding(
                config.pos_tag_vocab_size,
                config.hidden_size,
                padding_idx=0
            )

        self.graph_attention = []
        gat_out_size = head_dim * config.num_gat_head
        for i in range(config.num_gat_layer):
            input_size = config.hidden_size if i == 0 else gat_out_size
            self.graph_attention.append(
                GraphAttentionNetwork(
                    nfeat=input_size,
                    nhid=head_dim,
                    dropout=config.hidden_dropout_prob,  # 0.1
                    alpha=0.2,
                    nheads=config.num_gat_head
                )
            )
        self.graph_attention = nn.ModuleList(self.graph_attention)
        self.graph_attention_layer_norm = nn.LayerNorm(gat_out_size, eps=config.layer_norm_eps)

    def forward(
            self, inputs_embeds, dist_mat, deptag_ids=None, postag_ids=None
    ):
        gat_rep = inputs_embeds
        if self.dep_tag_embed is not None:
            assert deptag_ids is not None
            dep_embeddings = self.dep_tag_embed(deptag_ids)  # B x T x (L*head_dim)
            gat_rep += dep_embeddings
        if self.pos_tag_embed is not None:
            assert postag_ids is not None
            pos_embeddings = self.pos_tag_embed(postag_ids)  # B x T x (L*head_dim)
            gat_rep += pos_embeddings

        adj_mat = dist_mat.clone()
        adj_mat[dist_mat <= self.max_syntactic_distance] = 1
        adj_mat[dist_mat > self.max_syntactic_distance] = 0
        for _, gatlayer in enumerate(self.graph_attention):
            gat_rep = gatlayer(gat_rep, adj_mat)
        gat_rep = self.graph_attention_layer_norm(gat_rep)

        return gat_rep


class L1DistanceLoss(nn.Module):
    """Custom L1 loss for distance matrices."""

    def __init__(self, word_pair_dims=(1, 2)):
        super(L1DistanceLoss, self).__init__()
        self.word_pair_dims = word_pair_dims

    def forward(self, predictions, label_batch, length_batch):
        """ Computes L1 loss on distance matrices.
        Ignores all entries where label_batch=-1
        Normalizes first within sentences (by dividing by the square of the sentence length)
        and then across the batch.
        Args:
          predictions: A pytorch batch of predicted distances
          label_batch: A pytorch batch of true distances
          length_batch: A pytorch batch of sentence lengths
        Returns:
          A tuple of:
            batch_loss: average loss in the batch
            total_sents: number of sentences in the batch
        """
        labels_1s = (label_batch != 99999).float()
        predictions_masked = predictions * labels_1s
        labels_masked = label_batch * labels_1s
        total_sents = torch.sum((length_batch != 0)).float()
        squared_lengths = length_batch.pow(2).float()
        loss_per_sent = torch.sum(torch.abs(predictions_masked - labels_masked), dim=self.word_pair_dims)
        normalized_loss_per_sent = loss_per_sent / squared_lengths
        batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
        return batch_loss, total_sents


class L1DepthLoss(nn.Module):
    """Custom L1 loss for depth sequences."""

    def __init__(self, word_dim=1):
        super(L1DepthLoss, self).__init__()
        self.word_dim = word_dim

    def forward(self, predictions, label_batch, length_batch):
        """ Computes L1 loss on depth sequences.
        Ignores all entries where label_batch=-1
        Normalizes first within sentences (by dividing by the sentence length)
        and then across the batch.
        Args:
          predictions: A pytorch batch of predicted depths
          label_batch: A pytorch batch of true depths
          length_batch: A pytorch batch of sentence lengths
        Returns:
          A tuple of:
            batch_loss: average loss in the batch
            total_sents: number of sentences in the batch
        """
        total_sents = torch.sum(length_batch != 0).float()
        labels_1s = (label_batch != 99999).float()
        predictions_masked = predictions * labels_1s
        labels_masked = label_batch * labels_1s
        loss_per_sent = torch.sum(torch.abs(predictions_masked - labels_masked), dim=self.word_dim)
        normalized_loss_per_sent = loss_per_sent / length_batch.float()
        batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
        return batch_loss, total_sents


class Probe(nn.Module):
    pass


class TwoWordPSDProbe(Probe):
    """ Computes squared L2 distance after projection by a matrix.
    For a batch of sentences, computes all n^2 pairs of distances
    for each sentence in the batch.
    """

    def __init__(self, model_dim, probe_rank):
        super(TwoWordPSDProbe, self).__init__()
        self.proj = nn.Parameter(data=torch.zeros(model_dim, probe_rank))
        nn.init.uniform_(self.proj, -0.05, 0.05)

    def forward(self, batch):
        """ Computes all n^2 pairs of distances after projection
        for each sentence in a batch.
        Note that due to padding, some distances will be non-zero for pads.
        Computes (B(h_i-h_j))^T(B(h_i-h_j)) for all i,j
        Args:
          batch: a batch of word representations of the shape
            (batch_size, max_seq_len, representation_dim)
        Returns:
          A tensor of distances of shape (batch_size, max_seq_len, max_seq_len)
        """
        transformed = torch.matmul(batch, self.proj)
        batchlen, seqlen, rank = transformed.size()
        transformed = transformed.unsqueeze(2)
        transformed = transformed.expand(-1, -1, seqlen, -1)
        transposed = transformed.transpose(1, 2)
        diffs = transformed - transposed
        squared_diffs = diffs.pow(2)
        squared_distances = torch.sum(squared_diffs, -1)
        return squared_distances


class OneWordPSDProbe(Probe):
    """ Computes squared L2 norm of words after projection by a matrix."""

    def __init__(self, model_dim, probe_rank):
        super(OneWordPSDProbe, self).__init__()
        self.proj = nn.Parameter(data=torch.zeros(model_dim, probe_rank))
        nn.init.uniform_(self.proj, -0.05, 0.05)

    def forward(self, batch):
        """ Computes all n depths after projection
        for each sentence in a batch.
        Computes (Bh_i)^T(Bh_i) for all i
        Args:
          batch: a batch of word representations of the shape
            (batch_size, max_seq_len, representation_dim)
        Returns:
          A tensor of depths of shape (batch_size, max_seq_len)
        """
        transformed = torch.matmul(batch, self.proj)
        batchlen, seqlen, rank = transformed.size()
        norms = torch.bmm(transformed.view(batchlen * seqlen, 1, rank),
                          transformed.view(batchlen * seqlen, rank, 1))
        norms = norms.view(batchlen, seqlen)
        return norms
