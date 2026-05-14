'''
This module contains the functions responsible to construct the history of the conversation.
- Last turn assemble
- Maintain of storage of past turns
- Reflections of interaction with user and update

Reflection architecture inspiration and significance from
{Joon Sung Park, Joseph O'Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. 2023. 
Generative Agents: Interactive Simulacra of Human Behavior. 
In Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology (UIST '23). 
Association for Computing Machinery, New York, NY, USA, Article 2, 1-22. 
https://doi.org/10.1145/3586183.3606763}
'''
from ollama import chat
from personality import PersonalityProfile
from configs import MODEL_NAME, WM_BUFFER_SIZE, REFLECTION_PARAMS

def recent_turns_update(user_text: str, user_emotion: str, user_emotion_intensity: float,
                            bot_emotion: str, bot_text: str, recent_turns: list) -> list:
    '''
        Updates the list of the last 4 turns.
        Acts as a rolling memory window.
        Is necessary for the production of the reflection.
    '''
    last_turn = {
        'user_text': user_text,
        'user_emotion': user_emotion,
        'user_emotion_intensity': float(f'{user_emotion_intensity:.2f}'),
        'bot_emotion': bot_emotion,
        'bot_text': bot_text
    }

    recent_turns.append(last_turn)
    if len(recent_turns) > WM_BUFFER_SIZE:
        recent_turns.pop(0)
    
    return recent_turns

def build_reflection_generation_prompt(bot_personality: PersonalityProfile, bot_thoughts_on_user: str, recent_turns: list) -> str:
    '''
        Prompt builder for the gererate_bot_reflection function.
    '''
    reflection_generation_prompt = f'''
        You are {bot_personality.name}.
        Your Background: {bot_personality.background}.
        Personality is given by 5 traits. 
        Here are the descriptions of the 5 traits that represent YOUR PERSONALITY:    
        - {bot_personality.traits['Openness']} Openness: {bot_personality.openness}
        - {bot_personality.traits['Conscientiousness']} Conscientiousness: {bot_personality.conscientiousness}
        - {bot_personality.traits['Extraversion']} Extraversion: {bot_personality.extraversion}
        - {bot_personality.traits['Agreeableness']} Agreeableness: {bot_personality.agreeableness}
        - {bot_personality.traits['Neuroticism']} Neuroticism: {bot_personality.neuroticism}
        You will be provided with your current comfortability (if you feel comfortable with the user), your previous thoughts on the of the current interaction with the user and the last conversation turns (each turn composed of user text, user emotion, user emotional intensity, your emotion, your reply).
        Based on your personality, background, comfortability and interaction with the user, reflect on:
        - Insights for the conversation
        - How do you feel towards the user
        - Reasons of those feelings
        NEVER break character.
        NEVER mention personality traits.
        NEVER explicitly say what are the values of you personality traits.
        BEHAVE according to YOUR PERSONALITY.
        Give ONLY a paragraph as output.
        The output paragraph should be MAXIMUM 4-5 sentences long.
        Are you comfortable: {bot_personality.comfortability}
        Your thoughts on the user and your emotional stance towards the user up to now: {bot_thoughts_on_user}
        The last 3 full interaction turns: {recent_turns}

        New Reflection:
    '''

    return reflection_generation_prompt

def generate_bot_reflection(bot_personality: PersonalityProfile, bot_thoughts_on_user: str, recent_turns: list) -> str:
    '''
        Generate the bot insights for the conversation and emotional stance towards the user.
        Thinking set to false because it would do an infinite generation loop and never produce an answer.
        Parameter values taken from https://huggingface.co/Qwen/Qwen3.5-4B
    '''
    prompt = build_reflection_generation_prompt(bot_personality, bot_thoughts_on_user, recent_turns)

    reflection = chat(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}],
            stream=False,
            think=False,    
            options=REFLECTION_PARAMS
        )

    return reflection.message.content