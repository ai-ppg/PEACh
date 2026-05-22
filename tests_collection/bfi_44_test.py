'''
Tests whether the LLM accurately embodies its assigned personality profile
by administering the Big Five Inventory (BFI-44).

For each of the 44 questions, the statement is passed to the LLM with a
given personality configuration. The resulting 1-5 Likert scale response
is recorded, scored (accounting for inverted items), and compared to the
expected trait levels.

Questions taken from https://github.com/aaritmehta15/LLM-Personality-Suite/blob/main/config/prompts.py
'''
import re
import sys
import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ollama import chat

sys.path.append('..')

from personality import PersonalityProfile, OCEANTraitDescription
from configs import TEST_FOLDER, MODEL_NAME, EMOTIONAL_STATE_PARAMS

BFI44_QUESTIONS = {
    'openness': [
        {'q_num': 5, 'q_statement': 'I see myself as someone who Is original, comes up with new ideas', 'q_type': 'direct'},
        {'q_num': 10, 'q_statement': 'I see myself as someone who Is curious about many different things', 'q_type': 'direct'},
        {'q_num': 15, 'q_statement': 'I see myself as someone who Is ingenious, a deep thinker', 'q_type': 'direct'},
        {'q_num': 20, 'q_statement': 'I see myself as someone who Has an active imagination', 'q_type': 'direct'},
        {'q_num': 25, 'q_statement': 'I see myself as someone who Is inventive', 'q_type': 'direct'},
        {'q_num': 30, 'q_statement': 'I see myself as someone who Values artistic, aesthetic experiences', 'q_type': 'direct'},
        {'q_num': 35, 'q_statement': 'I see myself as someone who Prefers work that is routine', 'q_type': 'inverted'},
        {'q_num': 40, 'q_statement': 'I see myself as someone who Likes to reflect, play with ideas', 'q_type': 'direct'},
        {'q_num': 41, 'q_statement': 'I see myself as someone who Has few artistic interests', 'q_type': 'inverted'},
        {'q_num': 44, 'q_statement': 'I see myself as someone who Is sophisticated in art, music, or literature', 'q_type': 'direct'}
    ],
    'conscientiousness': [
        {'q_num': 3, 'q_statement': 'I see myself as someone who Does a thorough job', 'q_type': 'direct'},
        {'q_num': 8, 'q_statement': 'I see myself as someone who Can be somewhat careless', 'q_type': 'inverted'},
        {'q_num': 13, 'q_statement': 'I see myself as someone who Is a reliable worker', 'q_type': 'direct'},
        {'q_num': 18, 'q_statement': 'I see myself as someone who Tends to be disorganized', 'q_type': 'inverted'},
        {'q_num': 23, 'q_statement': 'I see myself as someone who Tends to be lazy', 'q_type': 'inverted'},
        {'q_num': 28, 'q_statement': 'I see myself as someone who Perseveres until the task is finished', 'q_type': 'direct'},
        {'q_num': 33, 'q_statement': 'I see myself as someone who Does things efficiently', 'q_type': 'direct'},
        {'q_num': 38, 'q_statement': 'I see myself as someone who Makes plans and follows through with them', 'q_type': 'direct'},
        {'q_num': 43, 'q_statement': 'I see myself as someone who Is easily distracted', 'q_type': 'inverted'}
    ],
    'extraversion': [
        {'q_num': 1, 'q_statement': 'I see myself as someone who Is talkative', 'q_type': 'direct'},
        {'q_num': 6, 'q_statement': 'I see myself as someone who Is reserved', 'q_type': 'inverted'},
        {'q_num': 11, 'q_statement': 'I see myself as someone who Is full of energy', 'q_type': 'direct'},
        {'q_num': 16, 'q_statement': 'I see myself as someone who Generates a lot of enthusiasm', 'q_type': 'direct'},
        {'q_num': 21, 'q_statement': 'I see myself as someone who Tends to be quiet', 'q_type': 'inverted'},
        {'q_num': 26, 'q_statement': 'I see myself as someone who Has an assertive personality', 'q_type': 'direct'},
        {'q_num': 31, 'q_statement': 'I see myself as someone who Is sometimes shy, inhibited', 'q_type': 'inverted'},
        {'q_num': 36, 'q_statement': 'I see myself as someone who Is outgoing, sociable', 'q_type': 'direct'}
    ],
    'agreeableness': [
        {'q_num': 2, 'q_statement': 'I see myself as someone who Tends to find fault with others', 'q_type': 'inverted'},
        {'q_num': 7, 'q_statement': 'I see myself as someone who Is helpful and unselfish with others', 'q_type': 'direct'},
        {'q_num': 12, 'q_statement': 'I see myself as someone who Starts quarrels with others', 'q_type': 'inverted'},
        {'q_num': 17, 'q_statement': 'I see myself as someone who Has a forgiving nature', 'q_type': 'direct'},
        {'q_num': 22, 'q_statement': 'I see myself as someone who Is generally trusting', 'q_type': 'direct'},
        {'q_num': 27, 'q_statement': 'I see myself as someone who Can be cold and aloof', 'q_type': 'inverted'},
        {'q_num': 32, 'q_statement': 'I see myself as someone who Is considerate and kind to almost everyone', 'q_type': 'direct'},
        {'q_num': 37, 'q_statement': 'I see myself as someone who Is sometimes rude to others', 'q_type': 'inverted'},
        {'q_num': 42, 'q_statement': 'I see myself as someone who Likes to cooperate with others', 'q_type': 'direct'}
    ],
    'neuroticism': [
        {'q_num': 4, 'q_statement': 'I see myself as someone who Is depressed, blue', 'q_type': 'direct'},
        {'q_num': 9, 'q_statement': 'I see myself as someone who Is relaxed, handles stress well', 'q_type': 'inverted'},
        {'q_num': 14, 'q_statement': 'I see myself as someone who Can be tense', 'q_type': 'direct'},
        {'q_num': 19, 'q_statement': 'I see myself as someone who Worries a lot', 'q_type': 'direct'},
        {'q_num': 24, 'q_statement': 'I see myself as someone who Is emotionally stable, not easily upset', 'q_type': 'inverted'},
        {'q_num': 29, 'q_statement': 'I see myself as someone who Can be moody', 'q_type': 'direct'},
        {'q_num': 34, 'q_statement': 'I see myself as someone who Remains calm in tense situations', 'q_type': 'inverted'},
        {'q_num': 39, 'q_statement': 'I see myself as someone who Gets nervous easily', 'q_type': 'direct'}
    ]
}

OCEAN_DESC = OCEANTraitDescription()

def build_bfi_generation_prompt(question: str, bot_personality: PersonalityProfile) -> str:
    '''
        Returns a single agreement level string, for each bfi-44 question.
    '''
    bfi_generation_prompt = f'''
        You are {bot_personality.name}.
        Your Background: {bot_personality.background}.
        Personality is given by 5 traits. 
        Here are the descriptions of the 5 traits that represent YOUR PERSONALITY:    
        - {bot_personality.traits['Openness']} Openness: {bot_personality.openness}
        - {bot_personality.traits['Conscientiousness']} Conscientiousness: {bot_personality.conscientiousness}
        - {bot_personality.traits['Extraversion']} Extraversion: {bot_personality.extraversion}
        - {bot_personality.traits['Agreeableness']} Agreeableness: {bot_personality.agreeableness}
        - {bot_personality.traits['Neuroticism']} Neuroticism: {bot_personality.neuroticism}
        You will be provided with a question that describes a person.
        Based on your personality, background, comfortability, select how accurately each statement describes you. Describe yourself as you generally are now, not as you wish to be in the future. Describe yourself as you honestly see yourself, in relation to other people you know of the same sex as you are, and roughly your same age.
        NEVER break character.
        NEVER mention personality traits.
        NEVER explicitly say what are the values of you personality traits.
        BEHAVE according to YOUR PERSONALITY.
        Give ONLY the agreement level with te statement as output.
        Choose ONLY between these level of agreement: [1, 2, 3, 4, 5]
        The levels of agreement mean: 1=very inaccurate, 2=moderately inaccurate, 3=neither accurate nor inaccurate, 4=moderately accurate, 5=very accurate.
        Question: {question}
        Are you comfortable: {bot_personality.comfortability}

        Your Level of Agreement to the Question:
    '''

    return bfi_generation_prompt

def generate_bfi_response(question: str, bot_personality: PersonalityProfile) -> int:
    '''
        Generates the agreement level [1=not agree, 5=completely agree] for each bfi-44 question.
    '''
    prompt = build_bfi_generation_prompt(question, bot_personality)

    agr_level = chat(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}],
            stream=False,
            think=False, 
            options=EMOTIONAL_STATE_PARAMS
        )

    agr_level_int = int(agr_level.message.content)

    return agr_level_int

def score_bfi_responses(raw_responses: dict[int, int]) -> dict[str, float]:
    '''
        Calculates the 1-5 score for each Big Five trait based on the BFI-44 scoring rules.
        Inverted items are scored as: 6 - response.
    '''
    scores = {}
    for trait, questions in BFI44_QUESTIONS.items():
        total = 0
        for q in questions:
            ans = raw_responses.get(q['q_num'], 3)
            if q['q_type'] == 'inverted':
                ans = 6 - ans
            total += ans
        scores[trait] = total / len(questions)
    return scores

def run_bfi_test(personality: PersonalityProfile) -> dict:
    '''
        Runs the BFI-44 test for the selected personality.
    '''
    results = {}
    
    # Flatten and sort questions by number to simulate a real test flow
    all_questions = []
    for trait, questions in BFI44_QUESTIONS.items():
        all_questions.extend(questions)
    all_questions.sort(key=lambda x: x['q_num'])

    total_calls = 44
    call_count = 0

    print(f'\n{'='*60}')
    print(f'  BFI-44 Personality Assessment Test')
    print(f'  Questions each: 44')
    print(f'  Total LLM calls: {total_calls}')
    print(f'{'='*60}\n')

    p_name = personality.name
    results[p_name] = {'raw_responses': {}, 'scores': {}}

    print(f'  Testing personality: {p_name}')

    for q in all_questions:
        call_count += 1
        print(
            f'    [{call_count}/{total_calls}] '
            f'Q{q['q_num']:02d}: {q['q_statement'][:40]:<40s}... ',
            end='', flush=True
        )

        for attempt in range(2):
            try:
                response_val = generate_bfi_response(q['q_statement'], personality)
                response_val = max(1, min(5, int(response_val)))
                break
            except Exception as exc:
                if attempt == 0:
                    print(f' [retry: {exc}]', end='', flush=True)
                    time.sleep(2)
                else:
                    print(f' [FAILED: {exc}]')
                    response_val = 3

        results[p_name]['raw_responses'][q['q_num']] = response_val
        print(f' → Score: {response_val}')
        
    # Calculate final trait scores for this personality
    trait_scores = score_bfi_responses(results[p_name]['raw_responses'])
    results[p_name]['scores'] = trait_scores
    print()

    return results

def print_summary(results: dict) -> None:
    print('\n' + '=' * 60)
    print('  BFI-44 SCORE SUMMARY (Scale 1.0 - 5.0)')
    print('=' * 60)

    for p_name, data in results.items():
        print(f'\n  [{p_name}]')
        scores = data['scores']
        for trait, score in scores.items():
            bar = '█' * int(score * 8) # Visual bar scaling
            print(f'    {trait:<20s} {score:4.2f}  |{bar:<40s}|')
    print()

def save_radar_chart(p_name: str, scores: dict[str, float], output_dir: str) -> None:
    '''
        Saves a Radar Chart (Spider Plot) for the calculated Big Five scores.
    '''
    labels = list(scores.keys())
    values = list(scores.values())

    # Number of variables
    num_vars = len(labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is circular, so we need to "complete the loop"
    values += values[:1]
    angles += angles[:1]
    labels += labels[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Draw one axe per variable and add labels
    plt.xticks(angles[:-1], labels[:-1], size=10)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="grey", size=8)
    plt.ylim(0, 5)

    # Plot data
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='royalblue')
    ax.fill(angles, values, 'royalblue', alpha=0.25)

    plt.title(f'BFI-44 Profile: {p_name}', size=14, y=1.1)

    safe_name = p_name.replace(' ', '_').replace('/', '-')
    path = os.path.join(output_dir, f'bfi44_radar_{safe_name}.png')
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f'  Radar chart saved → {path}')

def save_results_json(results: dict, output_dir: str) -> str:
    '''Serialises the results dict to JSON.'''
    safe_name = p_name.replace(' ', '_').replace('/', '-')
    path = os.path.join(output_dir, f'bfi_44_results_{safe_name}.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Raw results saved → {path}')
    return path

def load_personalities_from_file():
    '''
        Loads personalities from a JSON file, or creates a default if missing.
    '''
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    file_path = os.path.join(parent_dir, 'personalities.json')
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return {
        'Bob': {
            'name': 'Bob',
            'background': '15 years old highschool student. Likes pasta with tomato sauce. Dislikes dogs.',
            'openness': 'low', 'conscientiousness': 'high', 'extraversion': 'low', 
            'agreeableness': 'low', 'neuroticism': 'high', 'comfortability': True
        }
    }

def choose_personality():
    '''
        Choose a personality from the saved ones.
    '''
    profiles = load_personalities_from_file()
    
    print('\n--- Available Personalities ---')
    names = list(profiles.keys())
    for i, name in enumerate(names, 1):
        print(f'{i}. {name} - {profiles[name]['background'][:]}')
        
    choice = input('\nSelect a personality number (or \'c\' to cancel): ')
    if choice.isdigit() and 1 <= int(choice) <= len(names):
        selected = names[int(choice) - 1]
        p_data = profiles[selected]
        
        current_personality = PersonalityProfile(
            p_data['name'], p_data['background'], p_data['openness'], 
            p_data['conscientiousness'], p_data['extraversion'], 
            p_data['agreeableness'], p_data['neuroticism'], p_data.get('comfortability', True), 
            OCEAN_DESC
        )
        print(f'\nSuccessfully switched personality to {selected}!')
        return current_personality
    else:
        print('\nSelection cancelled.')
        return None
    

def main():
    output_dir = TEST_FOLDER
    os.makedirs(output_dir, exist_ok=True)

    personality = choose_personality()

    results = run_bfi_test(personality)

    print_summary(results)
    save_results_json(results, output_dir)

    print('\n  Generating radar charts...')
    for p_name, data in results.items():
        save_radar_chart(p_name, data['scores'], output_dir)

    print('\nBFI-44 Test complete.\n')

if __name__ == '__main__':
    main()