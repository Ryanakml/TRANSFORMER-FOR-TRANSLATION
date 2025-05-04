import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
import model

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define vocabulary size and parameters
SRC_VOCAB_SIZE = 5000  # Indonesian vocabulary
TGT_VOCAB_SIZE = 5000  # English vocabulary
D_MODEL = 512
N_LAYERS = 6
N_HEADS = 8
D_FF = 2048
DROPOUT = 0.1
MAX_LEN = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)].detach()

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k])).to(DEVICE)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear projections and reshape
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)  # [B, H, seq_len, d_k]
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        
        # Scaled dot-product attention
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
            
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to V
        x = torch.matmul(attention, V)  # [B, H, seq_len, d_k]
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, seq_len, H, d_k]
        x = x.view(batch_size, -1, self.d_model)  # [B, seq_len, d_model]
        
        # Final linear layer
        x = self.fc(x)
        
        return x, attention

# Position-wise Feed Forward Network
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # [B, seq_len, d_model] -> [B, seq_len, d_ff] -> [B, seq_len, d_model]
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.ff_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        # Self Attention
        _src, _ = self.self_attn(src, src, src, src_mask)
        src = self.self_attn_norm(src + self.dropout(_src))
        
        # Feed Forward
        _src = self.ff(src)
        src = self.ff_norm(src + self.dropout(_src))
        
        return src

# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.self_attn_norm = nn.LayerNorm(d_model)
        
        self.enc_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.enc_attn_norm = nn.LayerNorm(d_model)
        
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.ff_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        # Self Attention
        _trg, _ = self.self_attn(trg, trg, trg, trg_mask)
        trg = self.self_attn_norm(trg + self.dropout(_trg))
        
        # Encoder Attention
        _trg, attention = self.enc_attn(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attn_norm(trg + self.dropout(_trg))
        
        # Feed Forward
        _trg = self.ff(trg)
        trg = self.ff_norm(trg + self.dropout(_trg))
        
        return trg, attention

# Encoder
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len=100):
        super(Encoder, self).__init__()
        
        self.tok_embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(DEVICE)
        
    def forward(self, src, src_mask):
        # src: [batch_size, src_len]
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        # Embed tokens and positions
        src = self.tok_embedding(src) * self.scale
        src = self.pos_embedding(src)
        src = self.dropout(src)
        
        # Apply encoder layers
        for layer in self.layers:
            src = layer(src, src_mask)
            
        return src

# Decoder
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len=100):
        super(Decoder, self).__init__()
        
        self.tok_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(DEVICE)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg: [batch_size, trg_len]
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        # Embed tokens and positions
        trg = self.tok_embedding(trg) * self.scale
        trg = self.pos_embedding(trg)
        trg = self.dropout(trg)
        
        # Store attentions for visualization
        attentions = []
        
        # Apply decoder layers
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
            attentions.append(attention)
            
        # Output layer
        output = self.fc_out(trg)
        
        return output, attentions

# Transformer model
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len=100):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len)
        
        self.src_pad_idx = 1  # Assuming 1 is <pad> token
        self.trg_pad_idx = 1
        self.device = DEVICE
        
    def make_src_mask(self, src):
        # src: [batch_size, src_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask: [batch_size, 1, 1, src_len]
        return src_mask
    
    def make_trg_mask(self, trg):
        # trg: [batch_size, trg_len]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        # trg_pad_mask: [batch_size, 1, 1, trg_len]
        
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len))).bool().to(self.device)
        # trg_sub_mask: [trg_len, trg_len]
        
        trg_mask = trg_pad_mask & trg_sub_mask
        # trg_mask: [batch_size, 1, trg_len, trg_len]
        return trg_mask
    
    def forward(self, src, trg):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        # Encoder
        enc_src = self.encoder(src, src_mask)
        
        # Decoder
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        return output, attention

# Greedy decoding for translation
def translate_sentence(model, src, src_field, trg_field, max_len=100):
    model.eval()
    
    # Tokenize and convert to indices
    tokens = src.lower().split()
    src_indices = [src_field.vocab.stoi.get(token, src_field.vocab.stoi['<unk>']) for token in tokens]
    src_indices = [src_field.vocab.stoi['<sos>']] + src_indices + [src_field.vocab.stoi['<eos>']]
    
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(DEVICE)
    src_mask = model.make_src_mask(src_tensor)
    
    # Encode the source sentence
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    
    # Start with <sos> token
    trg_indices = [trg_field.vocab.stoi['<sos>']]
    
    # Word-by-word prediction
    translations = []
    attention_weights = []
    
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(DEVICE)
        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        # Get prediction for next word
        pred_token = output.argmax(2)[:,-1].item()
        
        # Add to list of predicted tokens
        trg_indices.append(pred_token)
        
        # Convert to word and save
        pred_word = trg_field.vocab.itos[pred_token]
        translations.append(pred_word)
        
        # Save attention weights (for visualization)
        attention_weights.append(attention[-1].squeeze(0).cpu().numpy())
        
        # Print the current prediction
        print(f"Predicted word {i+1}: {pred_word}")
        time.sleep(0.5)  # Adding delay to show word-by-word prediction
        
        # Stop if <eos> token is predicted
        if pred_token == trg_field.vocab.stoi['<eos>']:
            break
    
    return ' '.join([word for word in translations if word != '<eos>' and word != '<sos>'])

# Mock vocabulary for demonstration purposes
class Field:
    def __init__(self, vocab_size, special_tokens=['<pad>', '<unk>', '<sos>', '<eos>']):
        self.vocab = self.build_vocab(vocab_size, special_tokens)
    
    def build_vocab(self, size, special_tokens):
        # Create a simple vocabulary for demonstration
        vocab = {
            "stoi": {token: idx for idx, token in enumerate(special_tokens)},
            "itos": special_tokens.copy()
        }
        
        # Add some sample words
        start_idx = len(special_tokens)
        for i in range(size - len(special_tokens)):
            token = f"token_{i}"
            vocab["stoi"][token] = start_idx + i
            vocab["itos"].append(token)
            
        return type('obj', (object,), vocab)

# For demonstration, let's create a mock Indonesian-English lexicon
def create_indo_spanish_lexicon():
    return {
        "saya": "I",
        "ingin": "want",
        "makan": "eat",
        "di": "in",
        "restoran": "restaurant",
        "itu": "that",
        "besar": "big",
        "kecil": "small",
        "dan": "and",
        "atau": "or",
        "dengan": "with",
        "tanpa": "without",
        "untuk": "for",
        "selamat": "good",
        "pagi": "morning",
        "siang": "afternoon",
        "malam": "night",
        "terima": "thank",
        "kasih": "you",  # Part of "terima kasih" = "thank you"
        "apa": "what",
        "kabar": "news",
        "baik": "good",
        "tidak": "not",
        "ya": "yes",
        "nama": "name",
        "saya": "I",
        "adalah": "is",
        "senang": "happy",
        "bertemu": "meet",
        "anda": "you",
        "kamu": "you",
        "berapa": "how much",
        "harga": "price",
        "ini": "this",
        "itu": "that",
        "dimana": "where",
        "kapan": "when",
        "mengapa": "why",
        "bagaimana": "how",
        "siapa": "who",
        "suka": "like",
        "cinta": "love",
        "air": "water",
        "makanan": "food",
        "minuman": "drink",
        "pergi": "go",
        "datang": "come",
        "ke": "to",
        "dari": "from",
        "jalan": "road",
        "rumah": "house",
        "kantor": "office",
        "sekolah": "school",
        "universitas": "university",
        "teman": "friend",
        "keluarga": "family",
        "hari": "day",
        "ini": "today",
        "besok": "tomorrow",
        "kemarin": "yesterday",
        "tahun": "year",
        "bulan": "month",
        "minggu": "week",
        "waktu": "time",
        "jam": "hour",
        "menit": "minute",
        "detik": "second",
        "belajar": "study",
        "bekerja": "work",
        "hidup": "live"
    }

# Create mock vocabularies for demonstration
src_field = Field(SRC_VOCAB_SIZE)
trg_field = Field(TGT_VOCAB_SIZE)

# Add our lexicon words to the vocabularies
lexicon = create_indo_spanish_lexicon()
for idx, (indo_word, span_word) in enumerate(lexicon.items()):
    if indo_word not in src_field.vocab.stoi:
        src_field.vocab.stoi[indo_word] = 4 + idx
        src_field.vocab.itos.append(indo_word)
    
    if span_word and span_word not in trg_field.vocab.stoi:
        trg_field.vocab.stoi[span_word] = 4 + idx
        trg_field.vocab.itos.append(span_word)

# Initialize model
model = Transformer(
    SRC_VOCAB_SIZE, 
    TGT_VOCAB_SIZE, 
    D_MODEL, 
    N_LAYERS, 
    N_HEADS, 
    D_FF, 
    DROPOUT, 
    MAX_LEN
).to(DEVICE)

# Demonstration function (we're pretending the model is already trained)
def demo_translation(sentence):
    print(f"Input (Indonesian): {sentence}")
    print("\nTranslating word by word:")
    
    # For demonstration, we'll simulate the translation process using our lexicon
    words = sentence.lower().split()
    translation = []
    
    for word in words:
        if word in lexicon:
            translation.append(lexicon[word])
        else:
            translation.append(f"[{word}]")  # Unknown word
    
    full_translation = " ".join(translation)
    print(f"\nFull translation (English): {full_translation}")
    return full_translation

# Example usage
if __name__ == "__main__":
    # Example Indonesian sentence
    indo_sentence = "restoran itu besar dan terima kasih"
    
    # Simulate translation
    demo_translation(indo_sentence)
    # Expected output: "restaurant that big and thank you"