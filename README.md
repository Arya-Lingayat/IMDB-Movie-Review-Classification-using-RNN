# IMDB Movie Review Classification using RNN

This project implements a Recurrent Neural Network (RNN) model to perform sentiment analysis on the IMDB movie reviews dataset. The goal is to classify movie reviews as positive or negative based on their textual content.

## 📂 Repository Structure

- `main.py`: Script for training the RNN model on the IMDB dataset.
- `prediction.ipynb`: Jupyter notebook demonstrating how to make predictions using the trained model.
- `test.ipynb`: Notebook for testing and evaluating the model's performance.
- `simple_rnn_imdb.h5`: Saved weights of the trained RNN model.

## 📊 Dataset

The project utilizes the IMDB dataset, which contains 50,000 movie reviews labeled as positive or negative. The dataset is split evenly into 25,000 training and 25,000 testing samples. ([IMDB Review Sentiment Classification using RNN LSTM - KGP Talkie](https://kgptalkie.com/imdb-review-sentiment-classification-using-rnn-lstm/?utm_source=chatgpt.com))

## 🧠 Model Architecture

The model is built using Keras and consists of the following layers:

1. **Embedding Layer**: Converts word indices into dense vectors of fixed size.
2. **Simple RNN Layer**: Processes the sequence of embeddings to capture temporal dependencies.
3. **Dense Output Layer**: Applies a sigmoid activation to produce a binary classification output. ([GitHub - Gaurav2543/IMDB-Movie-Review-Sentiment-Analysis-using-RNNs-and-LSTMs: Developed a binary classifier for IMDB movie review dataset using RNNs and LSTMs](https://github.com/Gaurav2543/IMDB-Movie-Review-Sentiment-Analysis-using-RNNs-and-LSTMs?utm_source=chatgpt.com), [DL Student Lab Manual | PDF | Artificial Neural Network | Deep Learning](https://www.scribd.com/document/811576620/DL-Student-Lab-Manual?utm_source=chatgpt.com))

## 🛠️ Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Arya-Lingayat/IMDB-Movie-Review-Classification-using-RNN.git
   cd IMDB-Movie-Review-Classification-using-RNN
   ```


2. **Create a virtual environment (optional but recommended)**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```


3. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```


## 🚀 Usage

### Training the Model

To train the RNN model on the IMDB dataset, run:


```bash
python main.py
```


This will train the model and save the weights to `simple_rnn_imdb.h5`.

### Making Predictions

To make predictions on new data, open and run the `prediction.ipynb` notebook.
 
 
 
