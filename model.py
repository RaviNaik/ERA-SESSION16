import os
from typing import Any
import torch
import torch.nn as nn
import math

import lightning as L
import torchmetrics

import pandas as pd

from config import get_config
from dataset import causal_mask  # , TranslationDataModule

config = get_config()
# datamodule = TranslationDataModule(config)
# datamodule.setup()


class LayerNormalization(nn.Module):
    def __init__(self, eps=10**-6, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.FloatTensor):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, dropout: float, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.linear = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=d_ff, out_features=d_model),
        )

    def forward(self, x: torch.FloatTensor):
        return self.linear(x)


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x: torch.tensor):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(
        self, d_model: int, seq_len: int, dropout: float, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(1e4) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayers):
        return x + self.dropout(sublayers(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e4)

        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        value = self.w_v(v)
        key = self.w_k(k)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)

        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward_block)

        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = layers

        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block

        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )

        x = self.residual_connections[2](x, self.feed_forward_block)

        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.projection(x), dim=-1)


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


class TransformerModel(L.LightningModule):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_seq_len: int,
        tgt_seq_len: int,
        d_model: int,
        tokenizer_src,
        tokenizer_tgt,
        learning_rate: float,
        epochs: int,
        steps: int,
        max_lr: float = 10**-3,
        N: int = 6,
        h: int = 8,
        dropout: float = 0.1,
        d_ff: int = 2048,
    ) -> None:
        super().__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        self.d_model = d_model
        self.N = N
        self.h = h
        self.dropout = dropout
        self.d_ff = d_ff
        self.learning_rate = learning_rate
        self.max_lr = max_lr
        self.epochs = epochs
        self.steps = steps

        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt

        self.transformer = self.build_transformer()
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
        )

        self.cer_metric = torchmetrics.text.CharErrorRate()
        self.wer_metric = torchmetrics.text.WordErrorRate()
        self.bleu_metric = torchmetrics.text.BLEUScore()

        self.source_texts = []
        self.predicted = []
        self.expected = []

    def build_transformer(self):
        src_embed = InputEmbeddings(
            d_model=self.d_model, vocab_size=self.src_vocab_size
        )
        tgt_embed = InputEmbeddings(
            d_model=self.d_model, vocab_size=self.tgt_vocab_size
        )

        src_pos = PositionalEncoding(
            d_model=self.d_model, seq_len=self.src_seq_len, dropout=self.dropout
        )
        tgt_pos = PositionalEncoding(
            d_model=self.d_model, seq_len=self.tgt_seq_len, dropout=self.dropout
        )

        encoder_blocks = []
        for _ in range(self.N // 2):
            encoder_self_attention_block = MultiHeadAttentionBlock(
                d_model=self.d_model, h=self.h, dropout=self.dropout
            )

            feed_forward_block = FeedForwardBlock(
                d_model=self.d_model, d_ff=self.d_ff, dropout=self.dropout
            )

            encoder_block = EncoderBlock(
                self_attention_block=encoder_self_attention_block,
                feed_forward_block=feed_forward_block,
                dropout=self.dropout,
            )

            encoder_blocks.append(encoder_block)

        decoder_blocks = []
        for _ in range(self.N // 2):
            decoder_self_attention_block = MultiHeadAttentionBlock(
                d_model=self.d_model, h=self.h, dropout=self.dropout
            )

            decoder_cross_attention_block = MultiHeadAttentionBlock(
                d_model=self.d_model, h=self.h, dropout=self.dropout
            )

            feed_forward_block = FeedForwardBlock(
                d_model=self.d_model, d_ff=self.d_ff, dropout=self.dropout
            )

            decoder_block = DecoderBlock(
                self_attention_block=decoder_self_attention_block,
                cross_attention_block=decoder_cross_attention_block,
                feed_forward_block=feed_forward_block,
                dropout=self.dropout,
            )

            decoder_blocks.append(decoder_block)
        e1, e2, e3 = encoder_blocks
        d1, d2, d3 = decoder_blocks
        encoder = Encoder(nn.ModuleList([e1, e2, e3, e3, e2, e1]))
        decoder = Decoder(nn.ModuleList([d1, d2, d3, d3, d2, d1]))

        projection_layer = ProjectionLayer(
            d_model=self.d_model, vocab_size=self.tgt_vocab_size
        )

        transformer = Transformer(
            encoder=encoder,
            decoder=decoder,
            src_embed=src_embed,
            tgt_embed=tgt_embed,
            src_pos=src_pos,
            tgt_pos=tgt_pos,
            projection_layer=projection_layer,
        )

        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return transformer

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-9)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.max_lr,
            steps_per_epoch=self.steps,
            epochs=self.epochs,
            pct_start=1 / 10 if self.epochs != 1 else 0.5,
            div_factor=10,
            three_phase=True,
            final_div_factor=10,
            anneal_strategy="linear",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def forward(self, x):
        encoder_input = x["encoder_input"]
        decoder_input = x["decoder_input"]
        encoder_mask = x["encoder_mask"]
        decoder_mask = x["decoder_mask"]

        encoder_output = self.transformer.encode(encoder_input, encoder_mask)
        decoder_output = self.transformer.decode(
            encoder_output, encoder_mask, decoder_input, decoder_mask
        )
        proj_output = self.transformer.project(decoder_output)
        return proj_output

    def training_step(self, batch, batch_idx):
        label = batch["label"]
        proj_output = self(batch)

        loss = self.loss_fn(
            proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1)
        )
        self.log("train_loss", loss.item(), prog_bar=True)
        return loss

    def greedy_decode(self, source, source_mask, tokenizer_src, tokenizer_tgt, max_len):
        sos_idx = tokenizer_tgt.token_to_id("[SOS]")
        eos_idx = tokenizer_tgt.token_to_id("[EOS]")

        encoder_output = self.transformer.encode(source, source_mask)
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source)

        while True:
            if decoder_input.size(1) == max_len:
                break

            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask)

            out = self.transformer.decode(
                encoder_output, source_mask, decoder_input, decoder_mask
            )

            prob = self.transformer.project(out[:, -1])

            _, next_word = torch.max(prob, dim=1)

            decoder_input = torch.cat(
                [
                    decoder_input,
                    torch.empty(1, 1).type_as(source).fill_(next_word.item()),
                ],
                dim=1,
            )

            if next_word == eos_idx:
                break

        return decoder_input.squeeze(0)

    def validation_step(self, batch, batch_idx):
        encoder_input = batch["encoder_input"]
        encoder_mask = batch["encoder_mask"]
        assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
        model_out = self.greedy_decode(
            encoder_input,
            encoder_mask,
            self.tokenizer_src,
            self.tokenizer_tgt,
            self.tgt_seq_len,
        )

        source_text = batch["src_text"][0]
        target_text = batch["tgt_text"][0]
        model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy())

        self.source_texts.append(source_text)
        self.expected.append(target_text)
        self.predicted.append(model_out_text)

    def on_validation_epoch_end(self):
        cer = self.cer_metric(self.predicted, self.expected)
        wer = self.wer_metric(self.predicted, self.expected)
        bleu = self.bleu_metric(self.predicted, self.expected)

        self.log_dict({"cer": cer, "wer": wer, "bleu": bleu}, prog_bar=True)

        if not os.path.exists("results"):
            os.mkdir("results")

        df = pd.DataFrame(
            data={
                "Source": self.source_texts,
                "Expected": self.expected,
                "Predicted": self.predicted,
            }
        )

        df.to_csv(f"results/translation_{self.current_epoch}.csv")

        self.source_texts = []
        self.predicted = []
        self.expected = []
