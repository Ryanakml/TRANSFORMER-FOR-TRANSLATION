import torch
import torch.nn as nn
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import time
from model import (
    Transformer, Field, DEVICE, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
    D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT, MAX_LEN,
    create_indo_spanish_lexicon
)

# Set page configuration
st.set_page_config(page_title="Transformer Translation Visualizer", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
.main {padding: 1rem;}
.stApp {max-width: 1200px; margin: 0 auto;}
.visualization-container {margin-top: 2rem;}
.token-box {display: inline-block; padding: 0.5rem; margin: 0.25rem; border: 1px solid #ddd; border-radius: 4px;}
.token-box:hover {background-color: #f0f0f0;}
.attention-map {margin: 1rem 0;}
.step-container {margin: 1.5rem 0; padding: 1rem; border: 1px solid #eee; border-radius: 8px;}
.step-title {font-weight: bold; margin-bottom: 0.5rem;}
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("Transformer Translation Visualizer")
st.markdown("""
This interactive app visualizes how a Transformer model translates text from Indonesian to English. 
Explore each step of the translation process from tokenization to the final output.
""")

# Create a modified version of the translate_sentence function that exposes internal states
def visualize_translation(model, src_text, src_field, trg_field, max_len=100):
    model.eval()
    
    # Step 1: Tokenization
    tokens = src_text.lower().split()
    st.subheader("1. Tokenization")
    st.write("Input text is split into tokens:")
    token_cols = st.columns(min(len(tokens) + 2, 8))
    
    # Display tokens in a row
    with token_cols[0]:
        st.markdown("<div class='token-box' style='background-color: #e6f3ff;'>&lt;sos&gt;</div>", unsafe_allow_html=True)
    
    for i, token in enumerate(tokens):
        with token_cols[i+1 % len(token_cols)]:
            st.markdown(f"<div class='token-box'>{token}</div>", unsafe_allow_html=True)
    
    with token_cols[(len(tokens)+1) % len(token_cols)]:
        st.markdown("<div class='token-box' style='background-color: #e6f3ff;'>&lt;eos&gt;</div>", unsafe_allow_html=True)
    
    # Step 2: Convert tokens to indices
    st.subheader("2. Converting Tokens to Indices")
    src_indices = [src_field.vocab.stoi.get(token, src_field.vocab.stoi['<unk>']) for token in tokens]
    src_indices = [src_field.vocab.stoi['<sos>']] + src_indices + [src_field.vocab.stoi['<eos>']]
    
    # Display token-to-index mapping
    index_cols = st.columns(min(len(src_indices), 8))
    for i, idx in enumerate(src_indices):
        token = "<sos>" if i == 0 else "<eos>" if i == len(src_indices) - 1 else tokens[i-1]
        with index_cols[i % len(index_cols)]:
            st.markdown(f"<div class='token-box'>{token}<br><small>Index: {idx}</small></div>", unsafe_allow_html=True)
    
    # Create tensor
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(DEVICE)
    src_mask = model.make_src_mask(src_tensor)
    
    # Step 3: Embedding Lookup
    st.subheader("3. Embedding Lookup")
    st.write("Tokens are converted to dense vectors (embeddings):")
    
    # Get token embeddings
    with torch.no_grad():
        token_embeddings = model.encoder.tok_embedding(src_tensor) * model.encoder.scale
    
    # Visualize a sample of the embeddings (first 10 dimensions for first 5 tokens)
    fig, ax = plt.subplots(figsize=(10, 3))
    sample_embeddings = token_embeddings[0, :min(5, len(src_indices)), :10].cpu().numpy()
    sns.heatmap(sample_embeddings, cmap='viridis', ax=ax)
    ax.set_xlabel('Embedding Dimensions (showing first 10)')
    ax.set_ylabel('Tokens')
    ax.set_yticklabels(['<sos>'] + tokens[:min(4, len(tokens))])
    st.pyplot(fig)
    
    # Step 4: Positional Encoding
    st.subheader("4. Positional Encoding")
    st.write("Position information is added to embeddings:")
    
    # Get positional encodings
    with torch.no_grad():
        pos_encoding = model.encoder.pos_embedding.pe[0, :len(src_indices), :10].cpu().numpy()
    
    # Visualize positional encodings
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.heatmap(pos_encoding[:min(5, len(src_indices))], cmap='coolwarm', ax=ax)
    ax.set_xlabel('Encoding Dimensions (showing first 10)')
    ax.set_ylabel('Position')
    ax.set_yticklabels(range(min(5, len(src_indices))))
    st.pyplot(fig)
    
    # Step 5: Encoder Processing
    st.subheader("5. Encoder Processing")
    
    # Process through encoder
    with torch.no_grad():
        # Get embeddings with positional encoding
        src = token_embeddings
        src = model.encoder.pos_embedding(src)
        src = model.encoder.dropout(src)
        
        # Store intermediate encoder outputs
        encoder_outputs = [src.cpu().numpy()]
        encoder_attentions = []
        
        # Process through encoder layers
        for i, layer in enumerate(model.encoder.layers):
            # Get self-attention
            _src, attn = layer.self_attn(src, src, src, src_mask)
            src = layer.self_attn_norm(src + layer.dropout(_src))
            
            # Get feed-forward
            _src = layer.ff(src)
            src = layer.ff_norm(src + layer.dropout(_src))
            
            # Store outputs and attention
            encoder_outputs.append(src.cpu().numpy())
            encoder_attentions.append(attn.cpu().numpy())
        
        enc_src = src
    
    # Visualize encoder self-attention for the last layer
    if encoder_attentions:
        st.write("Encoder Self-Attention (Last Layer):")
        
        # Create tabs for different attention heads
        head_tabs = st.tabs([f"Head {i+1}" for i in range(min(N_HEADS, 4))])
        
        for h, tab in enumerate(head_tabs):
            with tab:
                fig, ax = plt.subplots(figsize=(7, 5))
                attn_data = encoder_attentions[-1][0, h, :len(src_indices), :len(src_indices)]
                sns.heatmap(attn_data, cmap='viridis', ax=ax, vmin=0, vmax=1)
                ax.set_xlabel('Key Tokens')
                ax.set_ylabel('Query Tokens')
                ax.set_xticklabels(['<sos>'] + tokens + ['<eos>'], rotation=45)
                ax.set_yticklabels(['<sos>'] + tokens + ['<eos>'])
                st.pyplot(fig)
    
    # Step 6: Decoder Processing (Autoregressive Generation)
    st.subheader("6. Autoregressive Decoding")
    st.write("The model generates the translation one token at a time:")
    
    # Start with <sos> token
    trg_indices = [trg_field.vocab.stoi['<sos>']]
    translations = []
    attention_weights = []
    decoder_outputs = []
    
    # Create a placeholder for the step-by-step generation
    generation_container = st.empty()
    
    # Word-by-word prediction
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(DEVICE)
        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            # Get token embeddings
            trg_embeddings = model.decoder.tok_embedding(trg_tensor) * model.decoder.scale
            trg_embeddings = model.decoder.pos_embedding(trg_embeddings)
            trg_embeddings = model.decoder.dropout(trg_embeddings)
            
            # Process through decoder layers
            trg = trg_embeddings
            layer_attentions = []
            
            for layer in model.decoder.layers:
                # Self attention
                _trg, self_attn = layer.self_attn(trg, trg, trg, trg_mask)
                trg = layer.self_attn_norm(trg + layer.dropout(_trg))
                
                # Encoder attention
                _trg, enc_attn = layer.enc_attn(trg, enc_src, enc_src, src_mask)
                trg = layer.enc_attn_norm(trg + layer.dropout(_trg))
                
                # Feed forward
                _trg = layer.ff(trg)
                trg = layer.ff_norm(trg + layer.dropout(_trg))
                
                # Store attention
                layer_attentions.append((self_attn.cpu().numpy(), enc_attn.cpu().numpy()))
            
            # Output layer
            output = model.decoder.fc_out(trg)
            decoder_outputs.append(output.cpu().numpy())
        
        # Get prediction for next word
        pred_token = output.argmax(2)[:,-1].item()
        
        # Add to list of predicted tokens
        trg_indices.append(pred_token)
        
        # Convert to word and save
        pred_word = trg_field.vocab.itos[pred_token]
        translations.append(pred_word)
        
        # Save attention weights (for visualization)
        if layer_attentions:
            attention_weights.append(layer_attentions[-1][1])  # Use encoder-decoder attention from last layer
        
        # Update the step-by-step generation display
        with generation_container.container():
            st.write(f"Step {i+1}: Generated token '{pred_word}'")
            
            # Show current translation
            current_translation = ' '.join([w for w in translations if w != '<eos>'])
            st.write(f"Translation so far: {current_translation}")
            
            # Show attention map for this step
            if attention_weights:
                st.write("Encoder-Decoder Attention (showing how the model focuses on input tokens):")
                fig, ax = plt.subplots(figsize=(10, 3))
                
                # Get attention for the current token (last position, first head)
                attn_map = attention_weights[-1][0, 0, -1, :len(src_indices)].squeeze()
                
                # Create a bar chart to show attention weights
                ax.bar(range(len(src_indices)), attn_map, color='skyblue')
                ax.set_xticks(range(len(src_indices)))
                ax.set_xticklabels(['<sos>'] + tokens + ['<eos>'], rotation=45)
                ax.set_ylabel('Attention Weight')
                ax.set_title(f"Attention for generating '{pred_word}'")
                st.pyplot(fig)
        
        # Simulate processing time
        time.sleep(0.5)
        
        # Stop if <eos> token is predicted
        if pred_token == trg_field.vocab.stoi['<eos>']:
            break
    
    # Step 7: Final Translation
    st.subheader("7. Final Translation")
    final_translation = ' '.join([word for word in translations if word != '<eos>' and word != '<sos>'])
    st.markdown(f"<div style='padding: 1rem; background-color: #f0f9ff; border-radius: 8px;'>"
                f"<strong>Input (Indonesian):</strong> {src_text}<br><br>"
                f"<strong>Translation (English):</strong> {final_translation}"
                f"</div>", unsafe_allow_html=True)
    
    # Step 8: Attention Visualization Summary
    st.subheader("8. Attention Visualization Summary")
    
    if attention_weights:
        # Create a comprehensive attention visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Combine all attention maps for a summary view (using the last layer)
        combined_attention = np.zeros((len(translations), len(src_indices)))
        for i, attn in enumerate(attention_weights):
            if i < len(translations):
                # Use the first head's attention
                combined_attention[i, :len(src_indices)] = attn[0, 0, -1, :len(src_indices)]
        
        # Create a heatmap
        sns.heatmap(combined_attention, cmap='viridis', ax=ax)
        ax.set_xlabel('Input Tokens')
        ax.set_ylabel('Output Tokens')
        ax.set_xticklabels(['<sos>'] + tokens + ['<eos>'], rotation=45)
        ax.set_yticklabels(translations, rotation=0)
        ax.set_title('Attention Weights Throughout Translation')
        st.pyplot(fig)
    
    return final_translation

# Create a lexicon for Indonesian to English (modified from Spanish)
def create_indo_english_lexicon():
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
        "adalah": "is",
        "senang": "happy",
        "bertemu": "meet",
        "anda": "you",
        "kamu": "you",
        "berapa": "how much",
        "harga": "price",
        "ini": "this",
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
eng_field = Field(TGT_VOCAB_SIZE)

# Add lexicon words to the vocabularies
lexicon = create_indo_spanish_lexicon()
for idx, (indo_word, eng_word) in enumerate(lexicon.items()):
    if indo_word not in src_field.vocab.stoi:
        src_field.vocab.stoi[indo_word] = 4 + idx
        src_field.vocab.itos.append(indo_word)
    
    if eng_word and eng_word not in eng_field.vocab.stoi:
        eng_field.vocab.stoi[eng_word] = 4 + idx
        eng_field.vocab.itos.append(eng_word)

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

# No need to add English words again as we've already done it above

# Main app interface
st.sidebar.header("Translation Settings")

# Input text area
user_input = st.sidebar.text_area("Enter Indonesian text:", 
                                 value="restoran itu besar dan terima kasih", 
                                 height=100)

# Add a button to trigger translation
if st.sidebar.button("Translate and Visualize"):
    if user_input.strip():
        # Perform the translation with visualization
        visualize_translation(model, user_input, src_field, eng_field)
    else:
        st.warning("Please enter some text to translate.")

# Add explanatory sections
with st.sidebar.expander("About the Transformer Model"):
    st.write("""
    The Transformer is a neural network architecture that uses self-attention mechanisms 
    to process sequential data. It was introduced in the paper 'Attention Is All You Need' 
    and has become the foundation for many state-of-the-art NLP models.
    
    Key components:
    - **Encoder**: Processes the input sequence
    - **Decoder**: Generates the output sequence
    - **Multi-Head Attention**: Allows the model to focus on different parts of the input
    - **Positional Encoding**: Adds information about token positions
    """)

with st.sidebar.expander("How to Use This App"):
    st.write("""
    1. Enter Indonesian text in the text area
    2. Click 'Translate and Visualize'
    3. Explore each step of the translation process
    4. Hover over visualizations for more details
    5. Watch how the model generates each token step by step
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<small>Created for educational purposes to visualize Transformer-based translation.</small>
""", unsafe_allow_html=True)

# Display a message when the app first loads
if not st.session_state.get('app_loaded', False):
    st.info("👈 Enter Indonesian text in the sidebar and click 'Translate and Visualize' to start.")
    st.session_state['app_loaded'] = True