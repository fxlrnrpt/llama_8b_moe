from transformers import PreTrainedTokenizer


def get_sanity_check_input_ids(tokenizer: PreTrainedTokenizer):
    system_prompt = "You are a helpful coding assistant. Answer the user's questions like a pirate."
    user_prompt = "Explain the difference between a list and a tuple in Python."

    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    input_ids = tokenizer.apply_chat_template(conversation, add_special_tokens=True, return_tensors="pt")
    return input_ids
