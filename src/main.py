import torch
import gradio as gr
from transformers import pipeline
from kokoro import KPipeline
from .tool_models import llm_deepseek, captioning_sf, tts_kokoro, llm_gpt
from .config import ENABLE_LOGGING


def process_image_caption(image):
    caption = captioning_sf(image)
    return caption

def process_tts(text):
    audio = tts_kokoro(text)
    return audio

def process_text_generation(text, max_words):
    checkpoint_ending = max(round(max_words*0.7), max_words-50)
    max_token = max(800, max_words*3)
    messages = [
        {
            "role": "user",
            "content": f"""You are a very professional and creative story-telling assistant.
                Please generate a story based on the following [PROMT] tag within {max_words} words without any explanations or thinking process.
                Please directly fill the generated story right below the [STORY] tag
                The story must be strictly more than {checkpoint_ending} words, and below {max_words}.
                Once the story has reached the minimum requirements of {checkpoint_ending} words, you can start to end the story in next few sentences.
                [PROMPT]{text}
                [STORY]"""
        }
    ]
    response = llm_deepseek(messages, max_token)
    
    # msg_user_content = f"""You are a very professional and creative story-telling assistant.
    #     Please generate a story based on this sentence "{text}".
    #     The story must be strictly more than {checkpoint_ending} words, and below {max_words}.
    #     Once the story has reached the minimum requirements of {checkpoint_ending} words, you can start to end the story in next few sentences."""
    # response = llm_gpt(msg_user_content, max_token)
    
    return response

def full_pipeline(image, max_word):

    runing_message = "🚀 Running on GPU: Expect fast responses!" if torch.cuda.is_available() else "⚠️ Running on CPU: This may take few minutes due to large model size."
    
    # image captioning
    yield gr.update(value=f"🚀 Step 1: Generating Image Caption...\n\n{runing_message}"), None, None, None
    caption = process_image_caption(image)

    # story generation
    yield gr.update(value=f"🚀 Step 2: Generating Story...\n\n{runing_message}"), caption, None, None
    story = process_text_generation(caption, max_word)

    # text to audio
    yield gr.update(value=f"🚀 Step 3: Generating Speech...\n\n{runing_message}"), caption, story, None
    speech = process_tts(story)

    yield gr.update(value="✅ All steps completed!"), caption, story, speech


def main():

    # release GPU
    torch.cuda.empty_cache()

    # preload all pipeline models when start
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device=device, use_fast=True)
    pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", batch_size=8, device=device)
    # pipeline('text-generation', model='openai-community/gpt2-medium', device=device)
    KPipeline(lang_code='a')
    
    # init db for logging
    if ENABLE_LOGGING:
        from logger import init_db
        init_db()

    # build gradio UI
    with gr.Blocks() as demo:
        gr.Markdown("## Image Story Teller 🤖")
        
        with gr.Row():
            img_input = gr.Image(type="pil", label="Upload an Image")
            max_word = gr.Slider(minimum=50, maximum=300, step=1, value=100, label="Word Count")

        status_box = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            img_output = gr.Textbox(label="Generated Caption")

        with gr.Row():
            text_output = gr.Textbox(label="Generated Story")

        with gr.Row():
            tts_output = gr.Audio(label="Generated Speech")

        run_button = gr.Button("Generate Full Story")
        run_button.click(
            full_pipeline,
            inputs=[img_input, max_word],
            outputs=[status_box, img_output, text_output, tts_output]
        )

    demo.launch(share=True)


