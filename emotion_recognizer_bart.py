'''
This script handles downloading and running EmoBART locally for emotion recognition.
It uses generative prompts to classify the user's emotion into a mapped subset, 
and then performs a second generation step to determine the intensity of that emotion.

This is using a simple model with 7 basic Ekman's emotions. A finer approach could use a GoEmotion recogniser,
that provides finer emotional states (has limitations), or VAD classification.

@article{liu2024emollms,
  title={EmoLLMs: A Series of Emotional Large Language Models and Annotation Tools for Comprehensive Affective Analysis},
  author={Liu, Zhiwei and Yang, Kailai and Zhang, Tianlin and Xie, Qianqian and Yu, Zeping and Ananiadou, Sophia},
  journal={arXiv preprint arXiv:2401.08508},
  year={2024}}
'''
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from configs import EMOBART_MODEL_ID, SAVE_EMOBART_DIR

def download_emobart_model():
    '''
        If the EmoBART model is not present, download from Hugging Face and save it.
    '''
    if not os.path.exists(os.path.join(SAVE_EMOBART_DIR, "config.json")):
        print(f"Downloading model {EMOBART_MODEL_ID} from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained(EMOBART_MODEL_ID)
        model = AutoModelForSeq2SeqLM.from_pretrained(EMOBART_MODEL_ID)
        
        tokenizer.save_pretrained(SAVE_EMOBART_DIR)
        model.save_pretrained(SAVE_EMOBART_DIR)
        print(f"Model saved successfully to {SAVE_EMOBART_DIR}!\n")
    else:
        print(f"Model already exists in '{SAVE_EMOBART_DIR}'. Skipping download...\n")


def get_emobart_tokenizer_and_model() -> tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    '''
        Returns tokenizer, model loaded from the local directory.
        Uses device_map='auto' to handle the large model size efficiently if GPUs are available.
    '''
    tokenizer = AutoTokenizer.from_pretrained(SAVE_EMOBART_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(SAVE_EMOBART_DIR, device_map='auto')

    return tokenizer, model


def get_user_emotional_status(tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM, 
                                user_text: str) -> tuple[str, float]:
    '''
        Gets user emotion mapped to a baseline set and generates an intensity score (0 to 1).
        return: user_emotion, emotion_intensity
    '''
    prompt_emo = f'''
        Task: Categorize the text's emotional tone as either 'neutral or no emotion' or identify the presence of one or more of the given emotions (anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, trust).
        Text: {user_text}
        This text contains emotions:
    '''
    
    input_ids_emo = tokenizer(prompt_emo, return_tensors='pt').input_ids.to(model.device)
    output_emo = model.generate(input_ids_emo, max_new_tokens=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    output_text_raw = tokenizer.decode(output_emo[0], skip_special_tokens=True)
    
    emotion_map = {     # had to do the map cause it would produce labels outide the prompted ones (cause of the finetuning made)
        'neutral': 'neutral',
        'no emotion': 'neutral',
        'anger': 'anger',
        'anticipation': 'neutral',
        'disgust': 'disgust',
        'fear': 'fear',
        'joy': 'joy',
        'love': 'neutral',
        'optimism': 'neutral',
        'pessimism': 'neutral',
        'sadness': 'sadness',
        'surprise': 'neutral',
        'trust': 'neutral'
    }

    predicted_emotions = [e.strip().lower() for e in output_text_raw.split(',')]
    mapped_emotion = 'neutral'
    for emotion in predicted_emotions:
        if emotion in emotion_map:
            mapped_emotion = emotion_map[emotion]
            break
        elif emotion == 'no emotion':
            mapped_emotion = 'neutral'
            break

    prompt_int = f'''
        Task: Assign a numerical value between 0 (least E) and 1 (most E) to represent the intensity of emotion E expressed in the text.
        Text: {user_text}
        Emotion: {mapped_emotion}
        Intensity Score:
    '''
    
    input_ids_int = tokenizer(prompt_int, return_tensors='pt').input_ids.to(model.device)
    output_int = model.generate(input_ids_int, max_new_tokens=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    emotion_intensity_str = tokenizer.decode(output_int[0], skip_special_tokens=True)
    
    try:
        emotion_intensity = float(emotion_intensity_str)
    except ValueError:
        emotion_intensity = emotion_intensity_str

    return mapped_emotion, emotion_intensity


if __name__ == '__main__':
    download_emobart_model()
    
    print("Loading model for test run...")
    tokenizer, model = get_emobart_tokenizer_and_model()
    
    test_text = input('You: ')
    emotion, intensity = get_user_emotional_status(tokenizer, model, test_text)
    
    print(f"Mapped Emotion: {emotion}")
    print(f"Intensity Score: {intensity}")