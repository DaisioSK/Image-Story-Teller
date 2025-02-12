from transformers import pipeline
import torch
import requests
from PIL import Image
from io import BytesIO
from kokoro import KPipeline
import soundfile as sf
import numpy as np
import time
from config import ENABLE_LOGGING
import utils


def tts_kokoro(text, download=0, output_format='wav'):

    pipeline = KPipeline(lang_code='a')
    generator = pipeline(
        text,
        voice='af_heart',
        speed=1,
        split_pattern=r'\n\n'
    )

    # concat tts audio clips
    audio_clips = [audio for (_, _, audio) in generator]
    full_audio = np.concatenate(audio_clips)
    output_file = f"output.{output_format}"
    sf.write(output_file, full_audio, 24000)

    return output_file


def llm_deepseek(messages, max_output_words=200):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    pipe = pipeline(
        "text-generation",
        model=model_name,
        device=device,
        batch_size=8,
    )

    temperature = 1 if max_output_words < 1000 else 0.7
    result = pipe(
        messages,
        max_new_tokens=max_output_words,
        return_full_text=True,
        temperature = temperature,
    )
    result_text = result[0]["generated_text"] # question and answer

    question = result_text[0]['content']
    answer = result_text[1]["content"]
    if "</think>" in answer:
        thinking = answer.split("</think>", 1)[0].strip()
        response = answer.split("</think>", 1)[-1].strip()
    else:
        thinking = ''
        response = answer
        
    if ENABLE_LOGGING:
        from logger import log_chat
        log_chat(question, response, thinking, max_output_words, utils.word_count(response))
        
    return response


def captioning_sf(image_source):
    """
    Generates a caption for an image using the Salesforce BLIP image captioning model.

    Args:
      image_source: The path to a local image file or a URL to an image, or a PIL.Image object.

    Returns:
      A string containing the generated caption, or an error message if the image cannot be processed.
    """

    try:
        # load BLIP model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device=device)

        if isinstance(image_source, str):
            if image_source.startswith("http"):
                # URL
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
                    "Referer": "https://www.sinchew.com.my/"
                }
                response = requests.get(image_source, headers=headers, stream=True)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            else:
                # local path
                image = Image.open(image_source)
        else:
            # image file
            image = image_source

        text = image_to_text(image)
        result = text[0]["generated_text"]
        print(f"{result}")

        return result

    except Exception as e:
        return f"An error occurred: {e}"


def llm_gpt(msg, max_output_words=200):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = pipeline('text-generation', model='gpt2-xl', device=device)
    response = generator(msg, max_new_tokens=max_output_words)
    print(response[0]['generated_text'])
