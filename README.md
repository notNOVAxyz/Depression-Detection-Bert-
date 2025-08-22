# Depression-Detection-Bert
This project is a text classification system using a fine-tuned BERT model to detect depression. It features a training script that downloads the dataset automatically and a prediction script for real-time, interactive analysis of text. The system is built with Hugging Face's transformers and datasets libraries.
# BERT-based Depression Detection System

This project is a text classification system designed to identify signs of depression in written text. It leverages a pre-trained **BERT** (Bidirectional Encoder Representations from Transformers) model, fine-tuned on a specialized dataset to classify sentences into one of two categories: **Depression** or **No Depression**.

The system is built using the **Hugging Face `transformers` and `datasets` libraries**, providing an efficient and powerful workflow for training and deployment.

---

## Key Features

* **State-of-the-Art Model**: Uses a pre-trained BERT model, which excels at understanding the context and nuance of human language.
* **Binary Text Classification**: The system is fine-tuned for a specific task, making it highly accurate at distinguishing between the two classes.
* **Efficient Workflow**: The training script automatically downloads the dataset and saves the fine-tuned model.
* **Interactive Prediction**: A separate prediction script allows for real-time testing of the model on new, user-provided text inputs.

---

## Prerequisites

To run this project, you need to have Python installed. It's recommended to set up a virtual environment and install the required libraries. You can use the following command to install all dependencies:

`pip install transformers torch pandas scikit-learn datasets`

---

## Getting Started

#### 1. Clone the Repository

Clone this repository to your local machine using the following command:

`git clone https://github.com/notNOVAxyz/Depression-Detection-Bert-/tree/main'

#### 2. Install Dependencies

Navigate into the project directory and install the required libraries.

`cd [Your Project Directory]`
`pip install -r requirements.txt`

#### 3. Train the Model

The `train_bert.py` script handles the entire training process. It will automatically download the dataset, fine-tune the BERT model with the optimal hyperparameters, and save the resulting model files to a new directory named `bert_depression_model`.

To train the model, run the following command in your terminal:

`python train_bert.py`

*Note: The model is configured to train for 10 epochs with a learning rate of 1.5e-5 and a batch size of 20, which were found to be effective for this dataset.*

#### 4. Make Predictions

After training is complete, the `predict.py` script can be used to test the model on new sentences. The script will load the saved model and enter an interactive loop.

Run the prediction script with this command:

`python predict.py`

---
