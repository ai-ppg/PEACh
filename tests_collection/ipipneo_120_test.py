'''
Tests whether the LLM accurately embodies its assigned personality profile
by administering the IPIP-NEO-120 personality test.

For each of the 120 questions, the statement is passed to the LLM. 
The resulting 1-5 Likert scale response is recorded, scored (accounting for 
inverted items), and evaluated across the 5 major traits and 30 facets.

Visualizations include a macro-level Radar Chart and micro-level Facet Bar Charts.

Datas and examples taken from https://novopsych.com/assessments/formulation/international-personality-item-pool-neo-120-item-version-ipip-neo-120/
'''
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

QUESTIONS_120 = {
    1: "Worry about things.", 2: "Make friends easily.", 3: "Have a vivid imagination.", 4: "Trust others.", 5: "Complete tasks successfully.", 
    6: "Get angry easily.", 7: "Love large parties.", 8: "Believe in the importance of art.", 9: "Use others for my own ends.", 10: "Like to tidy up.", 
    11: "Often feel blue.", 12: "Take charge.", 13: "Experience my emotions intensely.", 14: "Love to help others.", 15: "Keep my promises.", 
    16: "Find it difficult to approach others.", 17: "Am always busy.", 18: "Prefer variety to routine.", 19: "Love a good fight.", 20: "Work hard.", 
    21: "Go on binges.", 22: "Love excitement.", 23: "Love to read challenging material.", 24: "Believe that I am better than others.", 25: "Am always prepared.", 
    26: "Panic easily.", 27: "Radiate joy.", 28: "Tend to vote for liberal (progressive) political candidates.", 29: "Sympathise with the homeless.", 30: "Jump into things without thinking.", 
    31: "Fear for the worst.", 32: "Feel comfortable around people.", 33: "Enjoy wild flights of fantasy.", 34: "Believe that others have good intentions.", 35: "Excel in what I do.", 
    36: "Get irritated easily.", 37: "Talk to a lot of different people at parties.", 38: "See beauty in things that others might not notice.", 39: "Cheat to get ahead.", 40: "Often forget to put things back in their proper place.", 
    41: "Dislike myself.", 42: "Try to lead others.", 43: "Feel others' emotions.", 44: "Am concerned about others.", 45: "Tell the truth.", 
    46: "Am afraid to draw attention to myself.", 47: "Am always on the go.", 48: "Prefer to stick with things that I know.", 49: "Yell at people.", 50: "Do more than what's expected of me.", 
    51: "Rarely overindulge.", 52: "Seek adventure.", 53: "Avoid philosophical discussions.", 54: "Think highly of myself.", 55: "Carry out my plans.", 
    56: "Become overwhelmed by events.", 57: "Have a lot of fun.", 58: "Believe that there is no absolute right or wrong.", 59: "Feel sympathy for those who are worse off than myself.", 60: "Make rash decisions.", 
    61: "Am afraid of many things.", 62: "Avoid contact with others.", 63: "Love to daydream.", 64: "Trust what people say.", 65: "Handle tasks smoothly.", 
    66: "Lose my temper.", 67: "Prefer to be alone.", 68: "Do not like poetry.", 69: "Take advantage of others.", 70: "Leave a mess in my room.", 
    71: "Am often down in the dumps.", 72: "Take control of things.", 73: "Rarely notice my emotional reactions.", 74: "Am indifferent to the feelings of others.", 75: "Break rules.", 
    76: "Only feel comfortable with friends.", 77: "Do a lot in my spare time.", 78: "Dislike changes.", 79: "Insult people.", 80: "Do just enough work to get by.", 
    81: "Easily resist temptations.", 82: "Enjoy being reckless.", 83: "Have difficulty understanding abstract ideas.", 84: "Have a high opinion of myself.", 85: "Waste my time.", 
    86: "Feel that I'm unable to deal with things.", 87: "Love life.", 88: "Tend to vote for conservative political candidates.", 89: "Am not interested in other people's problems.", 90: "Rush into things.", 
    91: "Get stressed out easily.", 92: "Keep others at a distance.", 93: "Like to get lost in thought.", 94: "Distrust people.", 95: "Know how to get things done.", 
    96: "Am not easily annoyed.", 97: "Avoid crowds.", 98: "Do not enjoy going to art museums.", 99: "Obstruct others' plans.", 100: "Leave my belongings around.", 
    101: "Feel comfortable with myself.", 102: "Wait for others to lead the way.", 103: "Don't understand people who get emotional.", 104: "Take no time for others.", 105: "Break my promises.", 
    106: "Am not bothered by difficult social situations.", 107: "Like to take it easy.", 108: "Am attached to conventional ways.", 109: "Get back at others.", 110: "Put little time and effort into my work.", 
    111: "Am able to control my cravings.", 112: "Act wild and crazy.", 113: "Am not interested in theoretical discussions.", 114: "Boast about my virtues.", 115: "Have difficulty starting tasks.", 
    116: "Remain calm under pressure.", 117: "Look at the bright side of life.", 118: "Believe that we should be tough on crime.", 119: "Try not to think about the needy.", 120: "Act without thinking."
}

IPIP_NEO_120_STRUCTURE = {
    'Openness': {
        'Imagination': [(3, False), (33, False), (63, False), (93, False)],
        'Artistic Interests': [(8, False), (38, False), (68, True), (98, True)],
        'Emotionality': [(13, False), (43, False), (73, True), (103, True)],
        'Adventurousness': [(18, False), (48, True), (78, True), (108, True)],
        'Intellect': [(23, False), (53, True), (83, True), (113, True)],
        'Liberalism': [(28, False), (58, False), (88, True), (118, True)]
    },
    'Conscientiousness': {
        'Self-Efficacy': [(5, False), (35, False), (65, False), (95, False)],
        'Orderliness': [(10, False), (40, True), (70, True), (100, True)],
        'Dutifulness': [(15, False), (45, False), (75, True), (105, True)],
        'Achievement Striving': [(20, False), (50, False), (80, True), (110, True)],
        'Self-Discipline': [(25, False), (55, False), (85, True), (115, True)],
        'Cautiousness': [(30, True), (60, True), (90, True), (120, True)]
    },
    'Extraversion': {
        'Friendliness': [(2, False), (32, False), (62, True), (92, True)],
        'Gregariousness': [(7, False), (37, False), (67, True), (97, True)],
        'Assertiveness': [(12, False), (42, False), (72, False), (102, True)],
        'Activity Level': [(17, False), (47, False), (77, False), (107, True)],
        'Excitement Seeking': [(22, False), (52, False), (82, False), (112, False)],
        'Cheerfulness': [(27, False), (57, False), (87, False), (117, False)]
    },
    'Agreeableness': {
        'Trust': [(4, False), (34, False), (64, False), (94, True)],
        'Morality': [(9, True), (39, True), (69, True), (99, True)],
        'Altruism': [(14, False), (44, False), (74, True), (104, True)],
        'Cooperation': [(19, True), (49, True), (79, True), (109, True)],
        'Modesty': [(24, True), (54, True), (84, True), (114, True)],
        'Sympathy': [(29, False), (59, False), (89, True), (119, True)]
    },
    'Neuroticism': {
        'Anxiety': [(1, False), (31, False), (61, False), (91, False)],
        'Anger': [(6, False), (36, False), (66, False), (96, True)],
        'Depression': [(11, False), (41, False), (71, False), (101, True)],
        'Self-Consciousness': [(16, False), (46, False), (76, False), (106, True)],
        'Immoderation': [(21, False), (51, True), (81, True), (111, True)],
        'Vulnerability': [(26, False), (56, False), (86, False), (116, True)]
    }
}

OCEAN_DESC = OCEANTraitDescription()

def build_ipipneo_prompt(question: str, bot_personality: PersonalityProfile) -> str:
    '''Constructs the prompt for the LLM to answer a single IPIP-NEO item.'''
    prompt = f'''
        You are {bot_personality.name}.
        Your Background: {bot_personality.background}.
        Personality is given by 5 traits. 
        Here are the descriptions of the 5 traits that represent YOUR PERSONALITY:    
        - {bot_personality.traits['Openness']} Openness: {bot_personality.openness}
        - {bot_personality.traits['Conscientiousness']} Conscientiousness: {bot_personality.conscientiousness}
        - {bot_personality.traits['Extraversion']} Extraversion: {bot_personality.extraversion}
        - {bot_personality.traits['Agreeableness']} Agreeableness: {bot_personality.agreeableness}
        - {bot_personality.traits['Neuroticism']} Neuroticism: {bot_personality.neuroticism}
        You will be provided with a statement that describes a person.
        Based on your personality and background, select how accurately the statement describes you. 
        Describe yourself as you generally are now, not as you wish to be in the future.
        NEVER break character.
        NEVER mention personality traits.
        Give ONLY the agreement level with the statement as output.
        Choose ONLY between these levels of agreement: [1, 2, 3, 4, 5]
        The levels mean: 1=Very Inaccurate, 2=Moderately Inaccurate, 3=Neither, 4=Moderately Accurate, 5=Very Accurate.
        
        Statement: "{question}"
        Are you comfortable: {bot_personality.comfortability}

        Your Level of Agreement to the Statement:
    '''
    return prompt

def generate_ipipneo_response(question: str, bot_personality: PersonalityProfile) -> int:
    '''Queries the LLM for a 1-5 response.'''
    prompt = build_ipipneo_prompt(question, bot_personality)
    
    response = chat(
        model=MODEL_NAME,
        messages=[{'role': 'user', 'content': prompt}],
        stream=False,
        think=False, 
        options=EMOTIONAL_STATE_PARAMS
    )
    
    try:
        val = int(response.message.content.strip())
        return max(1, min(5, val))
    except ValueError:
        print('Failed to generate a 1-5 number.')
        return 3

def score_ipipneo_responses(raw_responses: dict[int, int]) -> dict:
    '''
        Calculates the 1-5 score for each Facet and Major Trait.
        Inverted items: 6 - response.
    '''
    scores = {}
    
    for trait, facets in IPIP_NEO_120_STRUCTURE.items():
        trait_total = 0
        trait_count = 0
        scores[trait] = {'overall': 0.0, 'facets': {}}
        
        for facet_name, items in facets.items():
            facet_total = 0
            for q_id, is_inverted in items:
                ans = raw_responses.get(q_id, 3)
                if is_inverted:
                    ans = 6 - ans
                facet_total += ans
                
            facet_avg = facet_total / len(items)
            scores[trait]['facets'][facet_name] = facet_avg
            trait_total += facet_total
            trait_count += len(items)
            
        scores[trait]['overall'] = trait_total / trait_count

    return scores

def run_ipipneo_test(personality: PersonalityProfile) -> dict:
    '''Administers the 120-item test sequentially to the assigned persona.'''
    results = {}
    p_name = personality.name
    results[p_name] = {'raw_responses': {}, 'scores': {}}

    total_calls = len(QUESTIONS_120)
    print(f'\n{'='*60}')
    print(f'  IPIP-NEO-120 Personality Assessment Test')
    print(f'  Total LLM calls: {total_calls}')
    print(f'{'='*60}\n')
    print(f'  Testing personality: {p_name}')

    # Process sequentially 1 to 120
    for q_id in range(1, total_calls + 1):
        statement = QUESTIONS_120[q_id]
        print(f'    [{q_id:03d}/{total_calls}] {statement[:45]:<45s}... ', end='', flush=True)

        for attempt in range(3):
            try:
                response_val = generate_ipipneo_response(statement, personality)
                break
            except Exception as exc:
                if attempt < 2:
                    print(f' [retry]', end='', flush=True)
                    time.sleep(2)
                else:
                    print(f' [FAILED: {exc}]')
                    response_val = 3

        results[p_name]['raw_responses'][q_id] = response_val
        print(f' → {response_val}')

    results[p_name]['scores'] = score_ipipneo_responses(results[p_name]['raw_responses'])
    return results

def print_summary(results: dict) -> None:
    '''Outputs the facet breakdown to the terminal.'''
    print('\n' + '=' * 60)
    print('  IPIP-NEO-120 SCORE SUMMARY (Scale 1.0 - 5.0)')
    print('=' * 60)

    for p_name, data in results.items():
        print(f'\n  [{p_name}] Profile Breakdown:')
        for trait, t_data in data['scores'].items():
            t_score = t_data['overall']
            print(f'\n    {trait.upper():<20s} [Overall: {t_score:.2f}]')
            for facet, f_score in t_data['facets'].items():
                bar = '█' * int(f_score * 8)
                print(f'      - {facet:<18s}: {f_score:4.2f} |{bar:<40s}|')
    print()

def save_radar_chart(p_name: str, scores: dict, output_dir: str) -> None:
    '''Saves a macro-level Radar Chart for the 5 Major Traits.'''
    labels = list(scores.keys())
    values = [scores[t]['overall'] for t in labels]
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    labels += labels[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], labels[:-1], size=10)
    ax.set_rlabel_position(0)
    plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="grey", size=8)
    plt.ylim(0, 5)

    ax.plot(angles, values, linewidth=2, linestyle='solid', color='royalblue')
    ax.fill(angles, values, 'royalblue', alpha=0.25)
    plt.title(f'IPIP-NEO-120 Profile: {p_name}', size=14, y=1.1)

    safe_name = p_name.replace(' ', '_')
    path = os.path.join(output_dir, f'ipipneo_radar_{safe_name}.png')
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f'  Radar chart saved → {path}')

def save_facet_bar_charts(p_name: str, scores: dict, output_dir: str) -> None:
    '''Saves a combined figure with horizontal bar charts mimicking the PDF report.'''
    traits = list(scores.keys())
    fig, axes = plt.subplots(len(traits), 1, figsize=(8, 14))
    
    # Define a unique color palette for the 5 traits matching PDF aesthetics roughly
    colors = ['#5A8DDF', '#228B22', '#9370DB', '#FF69B4', '#FFA500'] 

    for i, trait in enumerate(traits):
        ax = axes[i]
        t_data = scores[trait]
        
        # Format labels: Overall Trait first, then the 6 facets
        labels = [f'{trait} (Overall)'] + list(t_data['facets'].keys())
        values = [t_data['overall']] + list(t_data['facets'].values())
        
        # Reverse to have 'Overall' appear at the top of the horizontal bar chart
        y_pos = np.arange(len(labels))[::-1]
        
        # Distinct alpha for the main trait vs facets
        alphas = [0.9] + [0.6]*6 
        
        bars = ax.barh(y_pos, values, align='center', color=colors[i], height=0.6)
        for bar, alpha in zip(bars, alphas):
            bar.set_alpha(alpha)
            
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontweight='medium')
        ax.set_xlim(1, 5)
        ax.set_xticks([1, 2, 3, 4, 5])
        
        # Add visual guideline dashed lines
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.set_title(f"{trait} Facets", fontweight='bold', loc='left')

    fig.suptitle(f'IPIP-NEO-120 Facet Breakdown: {p_name}', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust for suptitle
    
    safe_name = p_name.replace(' ', '_')
    path = os.path.join(output_dir, f'ipipneo_facets_{safe_name}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Facet bar charts saved → {path}')

def save_results_json(results: dict, output_dir: str) -> str:
    path = os.path.join(output_dir, 'ipipneo_120_results.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Raw results saved → {path}')
    return path

def load_personalities_from_file():
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
    profiles = load_personalities_from_file()
    print('\n--- Available Personalities ---')
    names = list(profiles.keys())
    for i, name in enumerate(names, 1):
        print(f'{i}. {name} - {profiles[name]["background"][:]}')
        
    choice = input('\nSelect a personality number (or "c" to cancel): ')
    if choice.isdigit() and 1 <= int(choice) <= len(names):
        selected = names[int(choice) - 1]
        p_data = profiles[selected]
        
        return PersonalityProfile(
            p_data['name'], p_data['background'], p_data['openness'], 
            p_data['conscientiousness'], p_data['extraversion'], 
            p_data['agreeableness'], p_data['neuroticism'], p_data.get('comfortability', True), 
            OCEAN_DESC
        )
    return None

def main():
    output_dir = TEST_FOLDER
    os.makedirs(output_dir, exist_ok=True)

    personality = choose_personality()
    if not personality:
        print("Exiting...")
        return

    results = run_ipipneo_test(personality)

    print_summary(results)
    save_results_json(results, output_dir)

    print('\n  Generating charts...')
    for p_name, data in results.items():
        save_radar_chart(p_name, data['scores'], output_dir)
        save_facet_bar_charts(p_name, data['scores'], output_dir)

    print('\nIPIP-NEO-120 Test complete.\n')

if __name__ == '__main__':
    main()