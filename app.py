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
.step-container {margin: 1.5rem 0; padding: 1rem; border: 1px solid #eee; border-radius: 8px; background-color: #f9f9f9;}
.step-title {font-weight: bold; margin-bottom: 0.5rem; color: #1f77b4;}
.tooltip {position: relative; display: inline-block; border-bottom: 1px dotted #666; cursor: help;}
.tooltip .tooltiptext {visibility: hidden; width: 250px; background-color: #555; color: #fff; text-align: center; border-radius: 6px; padding: 10px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -125px; opacity: 0; transition: opacity 0.3s;}
.tooltip:hover .tooltiptext {visibility: visible; opacity: 1;}
.process-flow {padding: 15px; margin: 20px 0; background-color: #f0f7ff; border-radius: 8px;}
.highlight-box {background-color: #e6f3ff; padding: 10px; border-radius: 5px; margin: 10px 0;}
.concept-card {border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 10px 0; background-color: white;}
.concept-title {font-weight: bold; color: #2c3e50; margin-bottom: 8px;}
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("Transformer Translation Visualizer")

# Introduction with simplified explanation
st.markdown("""
## 🔍 Apa itu Transformer?

Transformer adalah jenis model AI yang sangat baik dalam memahami bahasa. Model ini digunakan untuk menerjemahkan teks dari satu bahasa ke bahasa lain.

Aplikasi ini memvisualisasikan bagaimana model Transformer menerjemahkan teks dari Bahasa Indonesia ke Bahasa Inggris, dengan menampilkan setiap langkah prosesnya secara interaktif.
""")

# Add a process flow diagram
st.markdown("""
<div class='process-flow'>
<h3>Proses Penerjemahan Transformer</h3>
<ol>
  <li><strong>Tokenisasi</strong> - Memecah kalimat menjadi kata-kata</li>
  <li><strong>Konversi ke Indeks</strong> - Mengubah kata menjadi angka yang dapat diproses</li>
  <li><strong>Embedding</strong> - Mengubah angka menjadi vektor padat</li>
  <li><strong>Encoding Posisi</strong> - Menambahkan informasi urutan kata</li>
  <li><strong>Encoder</strong> - Memproses teks input dengan mekanisme perhatian (attention)</li>
  <li><strong>Decoder</strong> - Menghasilkan terjemahan satu kata pada satu waktu</li>
  <li><strong>Output</strong> - Menampilkan hasil terjemahan lengkap</li>
</ol>
</div>
""", unsafe_allow_html=True)

# Add tooltip helper function
st.markdown("""
<div class='concept-card'>
  <div class='concept-title'>Konsep Penting</div>
  <p><span class='tooltip'>Attention<span class='tooltiptext'>Mekanisme yang memungkinkan model fokus pada bagian tertentu dari input saat menghasilkan output</span></span>: Bagaimana model "memperhatikan" kata-kata yang relevan</p>
  <p><span class='tooltip'>Token<span class='tooltiptext'>Unit dasar teks, biasanya kata atau bagian kata</span></span>: Potongan teks kecil (biasanya kata)</p>
  <p><span class='tooltip'>Embedding<span class='tooltiptext'>Representasi numerik dari kata yang menangkap makna semantiknya</span></span>: Representasi kata dalam bentuk angka</p>
  <p><span class='tooltip'>Encoder-Decoder<span class='tooltiptext'>Arsitektur yang memproses input (encoder) dan menghasilkan output (decoder)</span></span>: Bagian yang memproses input dan menghasilkan output</p>
</div>
""", unsafe_allow_html=True)

# Create a modified version of the translate_sentence function that exposes internal states
def visualize_translation(model, src_text, src_field, trg_field, max_len=100):
    model.eval()
    
    # Step 1: Tokenization
    tokens = src_text.lower().split()
    st.markdown("<div class='step-container'>", unsafe_allow_html=True)
    st.subheader("1. Tokenisasi")
    st.markdown("""
    <div class='highlight-box'>
    <strong>Apa itu Tokenisasi?</strong> Proses memecah kalimat menjadi kata-kata individual (token) agar dapat diproses oleh model.
    </div>
    """, unsafe_allow_html=True)
    st.write("Teks input dipecah menjadi token (kata-kata):")
    token_cols = st.columns(min(len(tokens) + 2, 8))
    
    # Display tokens in a row
    with token_cols[0]:
        st.markdown("<div class='token-box' style='background-color: #e6f3ff;'>&lt;sos&gt;</div>", unsafe_allow_html=True)
    
    for i, token in enumerate(tokens):
        with token_cols[i+1 % len(token_cols)]:
            st.markdown(f"<div class='token-box'>{token}</div>", unsafe_allow_html=True)
    
    with token_cols[(len(tokens)+1) % len(token_cols)]:
        st.markdown("<div class='token-box' style='background-color: #e6f3ff;'>&lt;eos&gt;</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Step 2: Convert tokens to indices
    st.markdown("<div class='step-container'>", unsafe_allow_html=True)
    st.subheader("2. Konversi Token ke Indeks")
    st.markdown("""
    <div class='highlight-box'>
    <strong>Mengapa Konversi ke Indeks?</strong> Komputer tidak dapat memproses kata secara langsung, sehingga setiap kata diubah menjadi angka unik dari kamus vocabulary.
    <br><br>
    <span class='tooltip'>SOS<span class='tooltiptext'>Start of Sentence - Token khusus yang menandakan awal kalimat</span></span> dan 
    <span class='tooltip'>EOS<span class='tooltiptext'>End of Sentence - Token khusus yang menandakan akhir kalimat</span></span> adalah token khusus yang ditambahkan.
    </div>
    """, unsafe_allow_html=True)
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
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Step 3: Embedding Lookup
    st.markdown("<div class='step-container'>", unsafe_allow_html=True)
    st.subheader("3. Embedding Lookup")
    st.markdown("""
    <div class='highlight-box'>
    <strong>Apa itu Embedding?</strong> Proses mengubah indeks token menjadi vektor padat yang menangkap makna semantik kata.
    Setiap kata direpresentasikan sebagai vektor dengan ratusan dimensi, yang memungkinkan model memahami hubungan antar kata.
    </div>
    """, unsafe_allow_html=True)
    st.write("Token diubah menjadi vektor padat (embeddings):")
    
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
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Step 4: Positional Encoding
    st.markdown("<div class='step-container'>", unsafe_allow_html=True)
    st.subheader("4. Encoding Posisi")
    st.markdown("""
    <div class='highlight-box'>
    <strong>Mengapa Encoding Posisi Penting?</strong> Transformer perlu tahu urutan kata dalam kalimat. 
    Encoding posisi menambahkan informasi tentang posisi setiap token dalam kalimat, sehingga model dapat memahami struktur kalimat.
    </div>
    """, unsafe_allow_html=True)
    st.write("Informasi posisi ditambahkan ke embeddings:")
    
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
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Step 5: Encoder Processing
    st.markdown("<div class='step-container'>", unsafe_allow_html=True)
    st.subheader("5. Pemrosesan Encoder")
    st.markdown("""
    <div class='highlight-box'>
    <strong>Apa yang Dilakukan Encoder?</strong> Encoder memproses semua token input secara bersamaan dan menghasilkan representasi kontekstual.
    Setiap token diperbarui berdasarkan hubungannya dengan semua token lain menggunakan mekanisme <span class='tooltip'>Self-Attention<span class='tooltiptext'>Mekanisme yang memungkinkan setiap token untuk memperhatikan token lain dalam kalimat yang sama</span></span>.
    </div>
    """, unsafe_allow_html=True)
    
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
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Step 6: Decoder Processing (Autoregressive Generation)
    st.markdown("<div class='step-container'>", unsafe_allow_html=True)
    st.subheader("6. Pemrosesan Decoder (Generasi Autoregressif)")
    st.markdown("""
    <div class='highlight-box'>
    <strong>Bagaimana Decoder Bekerja?</strong> Decoder menghasilkan terjemahan satu token pada satu waktu secara berurutan.
    Untuk setiap token baru, decoder memperhatikan semua token yang telah dihasilkan sebelumnya dan semua token input dari encoder.
    Proses ini disebut <span class='tooltip'>Autoregressive<span class='tooltiptext'>Proses menghasilkan output berdasarkan output sebelumnya</span></span> karena setiap prediksi bergantung pada prediksi sebelumnya.
    </div>
    """, unsafe_allow_html=True)
    st.write("Model menghasilkan terjemahan satu token pada satu waktu:")
    
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
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Step 7: Final Translation
    st.markdown("<div class='step-container'>", unsafe_allow_html=True)
    st.subheader("7. Hasil Terjemahan Akhir")
    st.markdown("""
    <div class='highlight-box'>
    <strong>Hasil Akhir:</strong> Setelah semua token dihasilkan (atau token EOS ditemukan), kita mendapatkan terjemahan lengkap.
    </div>
    """, unsafe_allow_html=True)
    final_translation = ' '.join([word for word in translations if word != '<eos>' and word != '<sos>'])
    st.markdown(f"<div style='padding: 1rem; background-color: #f0f9ff; border-radius: 8px;'>"
                f"<strong>Input (Bahasa Indonesia):</strong> {src_text}<br><br>"
                f"<strong>Terjemahan (Bahasa Inggris):</strong> {final_translation}"
                f"</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Step 8: Attention Visualization Summary
    st.markdown("<div class='step-container'>", unsafe_allow_html=True)
    st.subheader("8. Visualisasi Attention")
    st.markdown("""
    <div class='highlight-box'>
    <strong>Memahami Attention:</strong> Visualisasi ini menunjukkan bagaimana model memperhatikan kata-kata input saat menghasilkan setiap kata output.
    Warna yang lebih terang menunjukkan perhatian yang lebih tinggi. Ini membantu kita memahami bagaimana model membuat keputusan terjemahan.
    </div>
    """, unsafe_allow_html=True)
    
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
        ax.set_xlabel('Token Input')
        ax.set_ylabel('Token Output')
        ax.set_xticklabels(['<sos>'] + tokens + ['<eos>'], rotation=45)
        ax.set_yticklabels(translations, rotation=0)
        ax.set_title('Bobot Attention Selama Penerjemahan')
        st.pyplot(fig)
    
    st.markdown("</div>", unsafe_allow_html=True)  # Close the last step container
    
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
st.sidebar.header("Pengaturan Penerjemahan")

# Input text area with more descriptive label
st.sidebar.markdown("### Masukkan Teks Bahasa Indonesia")
st.sidebar.markdown("Ketik atau tempel teks Bahasa Indonesia yang ingin diterjemahkan:")
user_input = st.sidebar.text_area("", 
                                 value="restoran itu besar dan terima kasih", 
                                 height=100,
                                 key="input_text")

# Add examples for users to try
st.sidebar.markdown("### Contoh Kalimat")
if st.sidebar.button("Contoh 1: Sapaan"):
    st.session_state.input_text = "selamat pagi nama saya adalah andi"
    st.experimental_rerun()
    
if st.sidebar.button("Contoh 2: Makanan"):
    st.session_state.input_text = "saya ingin makan di restoran itu"
    st.experimental_rerun()
    
if st.sidebar.button("Contoh 3: Perjalanan"):
    st.session_state.input_text = "besok saya akan pergi ke kantor dengan teman"
    st.experimental_rerun()

# Add a button to trigger translation with more descriptive text
if st.sidebar.button("✨ Terjemahkan dan Visualisasikan", use_container_width=True):
    if user_input.strip():
        # Perform the translation with visualization
        visualize_translation(model, user_input, src_field, eng_field)
    else:
        st.warning("Silakan masukkan teks untuk diterjemahkan.")

# Add explanatory sections
with st.sidebar.expander("Tentang Model Transformer"):
    st.write("""
    Transformer adalah arsitektur jaringan saraf yang menggunakan mekanisme self-attention 
    untuk memproses data sekuensial. Model ini diperkenalkan dalam paper 'Attention Is All You Need' 
    dan telah menjadi dasar untuk banyak model NLP canggih saat ini.
    
    Komponen utama:
    - **Encoder**: Memproses urutan input
    - **Decoder**: Menghasilkan urutan output
    - **Multi-Head Attention**: Memungkinkan model fokus pada bagian berbeda dari input
    - **Positional Encoding**: Menambahkan informasi tentang posisi token
    """)

with st.sidebar.expander("Cara Menggunakan Aplikasi Ini"):
    st.write("""
    1. Masukkan teks Bahasa Indonesia di area teks atau pilih contoh
    2. Klik tombol '✨ Terjemahkan dan Visualisasikan'
    3. Jelajahi setiap langkah proses penerjemahan
    4. Arahkan kursor ke istilah bertitik untuk melihat penjelasan
    5. Amati bagaimana model menghasilkan setiap token secara bertahap
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<small>Dibuat untuk tujuan pendidikan untuk memvisualisasikan penerjemahan berbasis Transformer.</small>
""", unsafe_allow_html=True)

# Display a welcome message when the app first loads
if not st.session_state.get('app_loaded', False):
    st.markdown("""
    <div class='concept-card'>
        <div class='concept-title'>👋 Selamat Datang di Visualisasi Transformer!</div>
        <p>Aplikasi ini membantu Anda memahami cara kerja model Transformer dalam menerjemahkan teks.</p>
        <p>Untuk memulai:</p>
        <ol>
            <li>Masukkan teks Bahasa Indonesia di panel samping atau pilih salah satu contoh</li>
            <li>Klik tombol "✨ Terjemahkan dan Visualisasikan"</li>
            <li>Jelajahi setiap langkah proses penerjemahan</li>
        </ol>
        <p><strong>Tip:</strong> Arahkan kursor ke istilah yang bergaris bawah titik-titik untuk melihat penjelasan lebih detail!</p>
    </div>
    """, unsafe_allow_html=True)
    st.session_state['app_loaded'] = True