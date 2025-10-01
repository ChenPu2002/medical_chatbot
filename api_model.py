"""API model for OpenAI chat completions."""
import os
import openai
from dotenv import load_dotenv

load_dotenv('.env')
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()


def get_response(user_input, conversation_history, prompt_text):
    """Get response from OpenAI API."""
    # Note: prompt_text is currently unused but kept for API compatibility
    _ = prompt_text
    message = conversation_history + [{"role": "user", "content": user_input}]
    prompt = {
        "model": "gpt-3.5-turbo",
        "messages": message,
        "max_tokens": 150,
        "temperature": 0.7,  # Lower temperature to reduce randomness
        "frequency_penalty": 0.8,  # Penalize new topics to focus on discussion
        "presence_penalty": 0.8,  # Encourage taking the lead in conversation
        "stop": [" Human:", " AI:"],  # Use stopping conditions to manage output
    }

    completion = client.chat.completions.create(**prompt)
    response = completion.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": response})
    return response, conversation_history
