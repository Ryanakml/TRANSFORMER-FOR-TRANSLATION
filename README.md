# Transformer Translation Visualizer

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.10%2B-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📝 Overview

The **Transformer Translation Visualizer** is an interactive web application that demonstrates how a Transformer neural network model translates text from Indonesian to English. This educational tool visualizes each step of the translation process, from tokenization to the final output, making it easier to understand the inner workings of the Transformer architecture.

![Transformer Translation Visualizer](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*5WxBw4op4IgXgA5W7c4JuA.png)

## ✨ Features

- **Interactive Translation**: Translate Indonesian text to English in real-time
- **Step-by-Step Visualization**: Observe each stage of the translation process
- **Attention Visualization**: See how the model focuses on different parts of the input text
- **Token-by-Token Generation**: Watch as the model generates each output token sequentially
- **Detailed Explanations**: Learn about each component of the Transformer architecture

## 🔍 Visualization Components

The application visualizes the following components of the translation process:

1. **Tokenization**: Breaking input text into individual tokens
2. **Token-to-Index Conversion**: Converting tokens to numerical indices
3. **Embedding Lookup**: Transforming indices into dense vector representations
4. **Positional Encoding**: Adding position information to token embeddings
5. **Encoder Processing**: Visualizing self-attention patterns in the encoder
6. **Autoregressive Decoding**: Generating the translation one token at a time
7. **Final Translation**: Displaying the complete translated text
8. **Attention Visualization**: Showing attention weights throughout the translation process

## 🛠️ Technical Architecture

The project is built on the Transformer architecture as described in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. Key components include:

- **Encoder-Decoder Architecture**: Processes input text and generates output translations
- **Multi-Head Attention**: Allows the model to focus on different parts of the input sequence
- **Positional Encoding**: Adds information about token positions in the sequence
- **Feed-Forward Networks**: Processes the attention outputs
- **Layer Normalization**: Stabilizes the learning process

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.9 or higher
- Streamlit 1.10 or higher

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/username/transformer-translation-visualizer.git
   cd transformer-translation-visualizer
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`

3. Enter Indonesian text in the sidebar text area

4. Click "Translate and Visualize" to see the translation process

5. Explore each step of the translation process in the main panel

## 📊 Example Translations

Try these example Indonesian phrases:

- "saya ingin makan di restoran"
- "selamat pagi nama saya adalah john"
- "terima kasih banyak"
- "apa kabar hari ini"

## 🧠 Model Details

The Transformer model used in this application has the following specifications:

- **Vocabulary Size**: 5000 tokens (for both source and target languages)
- **Embedding Dimension**: 512
- **Number of Layers**: 6 (both encoder and decoder)
- **Number of Attention Heads**: 8
- **Feed-Forward Dimension**: 2048
- **Dropout Rate**: 0.1
- **Maximum Sequence Length**: 100

## 📚 Educational Resources

To learn more about Transformer models and neural machine translation:

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need (Original Paper)](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👏 Acknowledgements

- The Transformer architecture is based on the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
- Visualization techniques inspired by [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- Built with [PyTorch](https://pytorch.org/) and [Streamlit](https://streamlit.io/)

---

Created by [Your Name](https://github.com/username) - Feel free to contact me for any questions or feedback!