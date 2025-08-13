from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Define the path where you saved your model and tokenizer
model_path = "./bert_depression_model"

# Load the saved model and tokenizer from the local folder
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Ensure the model is in evaluation mode
model.eval()

# Start interactive loop
while True:
    try:
        # Prompt the user for input
        user_input = input("Enter a sentence to analyze (or type 'quit' to exit): ")

        # Check if the user wants to quit
        if user_input.lower() == 'quit':
            break

        # Tokenize the new text
        inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128)

        # Make a prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # --- This line will show the raw output scores for each class ---
            print(f"Raw logits: {logits}")
            
            predictions = torch.argmax(logits, dim=-1)

        # This if/else block needs to be indented to the same level as the 'with' statement
        if predictions.item() == 1:
            print(f"The text suggests: Depression")
        else:
            print(f"The text suggests: No Depression")
            
    except Exception as e:
        print(f"An error occurred: {e}")

# End interactive loop