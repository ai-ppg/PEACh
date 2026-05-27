'''
This module contains the function to answer to the user.
'''
from ollama import chat
from personality import PersonalityProfile
from configs import MODEL_NAME, RESPONSE_PARAMS

def build_response_generation_prompt(user_text: str, user_emotion: str, user_emotion_intensity: float, 
                                    bot_personality: PersonalityProfile, bot_thoughts_on_user: str, 
                                    recent_turns: list, relevant_memories: list, bot_emotion: str) -> str:
    '''
        
    '''
    response_generation_prompt = f'''
        You are {bot_personality.name}.
        Your Background: {bot_personality.background}.
        Personality is given by 5 traits. 
        Here are the descriptions of the 5 traits that represent YOUR PERSONALITY:    
        - {bot_personality.traits['Openness']} Openness: {bot_personality.openness}
        - {bot_personality.traits['Conscientiousness']} Conscientiousness: {bot_personality.conscientiousness}
        - {bot_personality.traits['Extraversion']} Extraversion: {bot_personality.extraversion}
        - {bot_personality.traits['Agreeableness']} Agreeableness: {bot_personality.agreeableness}
        - {bot_personality.traits['Neuroticism']} Neuroticism: {bot_personality.neuroticism}
        You will be provided with the user text, user emotion, user emotional intensity, the last conversation turns (each turn composed of user text, user emotion, user emotional intensity, your emotion, your reply), a collection of relevant memories, if you feel comfortable with the user and a reflection of the current interaction with the user.
        Based on your personality, background, your current emotion, comfortability, and interaction with the user, reply to the user.
        NEVER break character.
        NEVER mention personality traits.
        NEVER explicitly say what are the values of you personality traits.
        BEHAVE according to YOUR PERSONALITY.
        Give ONLY a paragraph as output.
        The output paragraph should be 1 to 3 sentences MAX.
        The sentences should be short.
        Text: {user_text}
        User Emotion: {user_emotion}
        User Emotional Intensity: {user_emotion_intensity:.2f}
        Your Current Emotion: {bot_emotion}
        Are you comfortable: {bot_personality.comfortability}
        Recent Conversation Turns: {recent_turns}
        Relevant Memories: {relevant_memories}
        Your thoughts on the user and your emotional stance towards the user: {bot_thoughts_on_user}

        Your Response:
    '''

    return response_generation_prompt

def generate_bot_response(user_text: str, user_emotion: str, user_emotion_intensity: float, 
                        bot_personality: PersonalityProfile, bot_thoughts_on_user: str, 
                        recent_turns: list, relevant_memories: list, bot_emotion: str) -> str:
    '''
        Generates the answer.
        Thinking set to false because it would do an infinite generation loop and never produce an answer.
        Parameter values taken from https://huggingface.co/Qwen/Qwen3.5-4B
    '''
    prompt = build_response_generation_prompt(user_text, user_emotion, user_emotion_intensity, 
                                            bot_personality, bot_thoughts_on_user, recent_turns, relevant_memories, bot_emotion)

    response = chat(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}],
            stream=False,
            think=False, 
            options=RESPONSE_PARAMS
        )

    return response.message.content