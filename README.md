# ERA-SESSION16 Transformer Implementation with OCP, PS & AMP in PytorchLightning

### Achieved:
1. **Training Loss: 1.740**
2. **CER Score: 0.526**
3. **BLEU Score: 0.000**
4. **WER Score: 0.969**

### Tasks:
1. :heavy_check_mark: Pick the "en-fr" dataset from opus_books
2. :heavy_check_mark: Remove all English sentences with more than 150 "tokens"
3. :heavy_check_mark: Remove all french sentences where len(fench_sentences) > len(english_sentrnce) + 10
4. :heavy_check_mark: Train your own transformer (E-D) (do anything you want, use PyTorch, OCP, PS, AMP, etc), but get your loss under 1.8

### Results
![image](https://github.com/RaviNaik/ERA-SESSION16/assets/23289802/0bd51449-88e5-4423-bc92-cdab57356ec3)
**Note:** Detailed results are presnt in results folder as a CSV file

### Model Summary
```python
 
   | Name                                    | Type               | Params
--------------------------------------------------------------------------------
0  | transformer                             | Transformer        | 68.1 M
1  | transformer.encoder                     | Encoder            | 9.4 M 
2  | transformer.encoder.layers              | ModuleList         | 9.4 M 
3  | transformer.encoder.norm                | LayerNormalization | 2     
4  | transformer.decoder                     | Decoder            | 12.6 M
5  | transformer.decoder.layers              | ModuleList         | 12.6 M
6  | transformer.decoder.norm                | LayerNormalization | 2     
7  | transformer.src_embed                   | InputEmbeddings    | 15.4 M
8  | transformer.src_embed.embedding         | Embedding          | 15.4 M
9  | transformer.tgt_embed                   | InputEmbeddings    | 15.4 M
10 | transformer.tgt_embed.embedding         | Embedding          | 15.4 M
11 | transformer.src_pos                     | PositionalEncoding | 0     
12 | transformer.src_pos.dropout             | Dropout            | 0     
13 | transformer.tgt_pos                     | PositionalEncoding | 0     
14 | transformer.tgt_pos.dropout             | Dropout            | 0     
15 | transformer.projection_layer            | ProjectionLayer    | 15.4 M
16 | transformer.projection_layer.projection | Linear             | 15.4 M
17 | loss_fn                                 | CrossEntropyLoss   | 0     
18 | cer_metric                              | CharErrorRate      | 0     
19 | wer_metric                              | WordErrorRate      | 0     
20 | bleu_metric                             | BLEUScore          | 0     
--------------------------------------------------------------------------------
68.1 M    Trainable params
0         Non-trainable params
68.1 M    Total params
272.582   Total estimated model params size (MB)
```
### Restricting English tokens to >= 150 and French tokens <= len(English Tokens)+10
```python
def process_dataset(ds: HFDataset, config):
    cleaned_data = []
    for item in ds:
        source = item["translation"][config["lang_src"]]
        target = item["translation"][config["lang_tgt"]]
        source_tokens = re.findall(r"\w+|[^\w\s]+", source)
        target_tokens = re.findall(r"\w+|[^\w\s]+", target)
        if len(source_tokens) > 150:
            continue
        if len(target_tokens) > len(source_tokens) + 10:
            continue
        cleaned_data.append({"id": item["id"], "translation": item["translation"]})
    return cleaned_data
```
**Output:**
![image](https://github.com/RaviNaik/ERA-SESSION16/assets/23289802/ef9fa30d-5fc9-46c7-8a50-3aa465d317d9)

### Parameter Sharing
```python
e1, e2, e3 = encoder_blocks
d1, d2, d3 = decoder_blocks
encoder = Encoder(nn.ModuleList([e1, e2, e3, e3, e2, e1]))
decoder = Decoder(nn.ModuleList([d1, d2, d3, d3, d2, d1]))
```
### Loss & Other Metrics
**Training Loss:**
![image](https://github.com/RaviNaik/ERA-SESSION16/assets/23289802/476c27ff-b38a-4c78-a457-449960030c83)


**CER, WER & BLEU Score:**
![image](https://github.com/RaviNaik/ERA-SESSION16/assets/23289802/4ce898a7-b507-4274-abd2-689793da1242)

### Tensorboard Plots 
