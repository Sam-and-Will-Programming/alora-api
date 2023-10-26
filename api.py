from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load the pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

@app.route("/chatbot", methods=["POST"])
def chatbot():
    user_input = request.json["user_input"]
    input_ids = tokenizer.encode(user_input, return_tensors="pt")

    # Generate a response from the chatbot
    response_ids = model.generate(input_ids, max_length=50, num_return_sequences=1, pad_token_id=50256)
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run()
