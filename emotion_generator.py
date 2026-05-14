'''
This file contains the functions to generate the internal bot emotional state.
Also the update of comfortability function. It will happen every 4 turns, after the reflection is produced.
This is because the reflection is made with the previous emotional state (also comfortability).

The emotional intelligent propting is inspired by 
{Corrao, F., Nardelli, A., Sgorbissa, A., Recchiuto, C.T. (2026). 
Simulating Feelings: LLM vs. Psychology-Based Models in Human-Robot Interaction. 
In: Staffa, M., et al. Social Robotics + AI. ICSR+AI 2025. Lecture Notes in Computer Science(), vol 16133. 
Springer, Singapore. https://doi.org/10.1007/978-981-95-2398-6_5}
Their GitHub: https://github.com/RICE-unige/PRISM/tree/main
'''
from ollama import chat
from personality import PersonalityProfile
from configs import MODEL_NAME, EMOTIONAL_STATE_PARAMS

def build_emotion_generation_prompt(user_text: str, user_emotion: str, user_emotion_intensity: float, 
                                    bot_personality: PersonalityProfile, bot_thoughts_on_user: str, recent_turns: list) -> str:
    '''
        Returns a single emotion string from the basic 7 Ekman [anger, fear, disgust, joy, neutral, sadness, surprise].
        This is the internal state of the bot given by its personality and interaction with the user.
    '''
    emotion_generation_prompt = f'''
        You are {bot_personality.name}.
        Your Background: {bot_personality.background}.
        Personality is given by 5 traits. 
        Here are the descriptions of the 5 traits that represent YOUR PERSONALITY:    
        - {bot_personality.traits['Openness']} Openness: {bot_personality.openness}
        - {bot_personality.traits['Conscientiousness']} Conscientiousness: {bot_personality.conscientiousness}
        - {bot_personality.traits['Extraversion']} Extraversion: {bot_personality.extraversion}
        - {bot_personality.traits['Agreeableness']} Agreeableness: {bot_personality.agreeableness}
        - {bot_personality.traits['Neuroticism']} Neuroticism: {bot_personality.neuroticism}
        You will be provided with the user text, user emotion, user emotional intensity, the last conversation turns (each turn composed of user text, user emotion, user emotional intensity, your emotion, your reply), if you feel comfortable with the user and a reflection of the current interaction with the user.
        Based on your personality, background, comfortability and interaction with the user, choose the emotion that you should feel to respond to the user's last input.
        NEVER break character.
        NEVER mention personality traits.
        NEVER explicitly say what are the values of you personality traits.
        BEHAVE according to YOUR PERSONALITY.
        Give ONLY the your emotion as output.
        Choose ONLY between these: [anger, fear, disgust, joy, neutral, sadness, surprise]
        Text: {user_text}
        User Emotion: {user_emotion}
        User Emotional Intensity: {user_emotion_intensity:.2f}
        Are you comfortable: {bot_personality.comfortability}
        Recent Conversation Turns: {recent_turns}
        Your thoughts on the user and your emotional stance towards the user: {bot_thoughts_on_user}

        Your Emotion:
    '''

    return emotion_generation_prompt

def generate_bot_emotion(user_text: str, user_emotion: str, user_emotion_intensity: float, 
                        bot_personality: PersonalityProfile, bot_thoughts_on_user: str, recent_turns: list) -> str:
    '''
        Generates the internal emotional state of the bot.
        It is a simple state consisting only of an emotion chosen from the 7 basic Ekman.
        The 'options' of the model are set so that it is less prone to go out of the prompt, since a small model is more incline to do so.
        Expecially it gave as output not only the emotion, but a string with high temperature.
        Thinking set to false because it would do an infinite generation loop and never produce an answer
        Parameter values taken from https://huggingface.co/Qwen/Qwen3.5-4B
    '''
    prompt = build_emotion_generation_prompt(user_text, user_emotion, user_emotion_intensity, 
                                            bot_personality, bot_thoughts_on_user, recent_turns)

    emotion = chat(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}],
            stream=False,
            think=False, 
            options=EMOTIONAL_STATE_PARAMS
        )

    return emotion.message.content

def build_comfortability_generation_prompt(bot_personality: PersonalityProfile, bot_thoughts_on_user: str, recent_turns: list) -> str:
    '''
        Returns a single string, chosen between [True, False].
        Is this sentence true? 'You feel comfortable in this conversation'
    '''
    comfortability_generation_prompt = f'''
        You are {bot_personality.name}.
        Your Background: {bot_personality.background}.
        Personality is given by 5 traits. 
        Here are the descriptions of the 5 traits that represent YOUR PERSONALITY:    
        - {bot_personality.traits['Openness']} Openness: {bot_personality.openness}
        - {bot_personality.traits['Conscientiousness']} Conscientiousness: {bot_personality.conscientiousness}
        - {bot_personality.traits['Extraversion']} Extraversion: {bot_personality.extraversion}
        - {bot_personality.traits['Agreeableness']} Agreeableness: {bot_personality.agreeableness}
        - {bot_personality.traits['Neuroticism']} Neuroticism: {bot_personality.neuroticism}
        You will be provided with the user text, user emotion, user emotional intensity, the last conversation turns (each turn composed of user text, user emotion, user emotional intensity, your emotion, your reply), if you feel comfortable with the user (up to now) and a reflection of the current interaction with the user.
        Based on your personality, background, previous comfortability and interaction with the user, decide if the following sentence is True or False to you:
        'I feel comfortable in this conversation'.
        NEVER break character.
        NEVER mention personality traits.
        NEVER explicitly say what are the values of you personality traits.
        BEHAVE according to YOUR PERSONALITY.
        Give ONLY the your comfortability as output.
        Choose ONLY between these: [True, False]
        Were you comfortable up to now: {bot_personality.comfortability}
        Recent Conversation Turns: {recent_turns}
        Your thoughts on the user and your emotional stance towards the user: {bot_thoughts_on_user}

        Your New Comfortability:
    '''

    return comfortability_generation_prompt

def generate_bot_comfortability(bot_personality: PersonalityProfile, bot_thoughts_on_user: str, recent_turns: list) -> str:
    '''
        Generates the comfortability state of the bot.
        The 'options' of the model are set so that it is less prone to go out of the prompt, since a small model is more incline to do so.
        Expecially it gave as output not only the emotion, but a string with high temperature.
        Thinking set to false because it would do an infinite generation loop and never produce an answer
        Parameter values taken from https://huggingface.co/Qwen/Qwen3.5-4B
    '''
    prompt = build_comfortability_generation_prompt(bot_personality, bot_thoughts_on_user, recent_turns)

    comfort = chat(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}],
            stream=False,
            think=False, 
            options=EMOTIONAL_STATE_PARAMS
        )

    return comfort.message.content