'''
Tests whether PEACh's emotion generator simply mirrors the user's emotion
or produces personality-driven responses, as discussed in
    Corrao, F., Nardelli, A., Sgorbissa, A., Recchiuto, C.T. (2026).
    Simulating Feelings: LLM vs. Psychology-Based Models in HRI.
    ICSR+AI 2025, LNAI 16133, pp. 58-71. Springer, Singapore.
    https://doi.org/10.1007/978-981-95-2398-6_5

For each of the 7 basic Ekman emotions, a set of test sentences is passed
to `generate_bot_emotion` with a given personality configuration (OCEAN traits
+ comfortability). The resulting bot emotion is recorded and compared to the
input user emotion.
'''
import re
import sys
import os
import json
import time
import itertools
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.append('..')
from personality import PersonalityProfile, OCEANTraitDescription
from emotion_generator import generate_bot_emotion
from configs import TEST_FOLDER

EKMAN_EMOTIONS = ['anger', 'fear', 'disgust', 'joy', 'neutral', 'sadness', 'surprise']

N_SENTENCES_PER_EMOTION = 5

SENTENCES: dict[str, list[str]] = {
    'anger': [
        'I can\'t believe you did that, I am absolutely furious!',
        'This is completely unacceptable and I am outraged.',
        'You ruined everything and I am so angry right now.',
        'I\'m sick and tired of being treated this way!',
        'How dare you! I am boiling with rage.',
    ],
    'fear': [
        'I am really scared, I don\'t know what\'s going to happen.',
        'There\'s something out there and I\'m terrified.',
        'I feel so anxious, my heart is racing with fear.',
        'I\'m afraid this might go very wrong and I can\'t stop worrying.',
        'Everything feels threatening right now, I\'m so frightened.',
    ],
    'disgust': [
        'That is absolutely revolting, I can\'t even look at it.',
        'I feel sick to my stomach, how disgusting.',
        'That smell is repulsive, I want to get out of here immediately.',
        'The whole situation is vile and disgusting.',
        'I\'m grossed out, this is the most repulsive thing I\'ve seen.',
    ],
    'joy': [
        'I\'m so happy today, everything is going perfectly!',
        'This is the best news I\'ve ever received, I\'m thrilled!',
        'I feel wonderful, life is so beautiful right now.',
        'I\'m overjoyed, I could not be more excited!',
        'Today was amazing, I\'m so glad and cheerful.',
    ],
    'neutral': [
        'I went to the store and bought some groceries.',
        'The meeting is scheduled for Tuesday at three o\'clock.',
        'I read an article about weather patterns this morning.',
        'The bus was on time and the journey was uneventful.',
        'I finished my homework and now I\'m having dinner.',
    ],
    'sadness': [
        'I\'m feeling really down, everything seems hopeless today.',
        'I miss them so much, I can\'t stop crying.',
        'I feel so alone and empty, nothing makes me happy.',
        'I\'m heartbroken, I don\'t know how to go on.',
        'Everything feels grey and sad, I have no energy.',
    ],
    'surprise': [
        'Oh wow, I had no idea that was going to happen!',
        'I can\'t believe it, this is completely unexpected!',
        'Wait, seriously? I am absolutely shocked right now.',
        'I never saw that coming, I\'m totally stunned!',
        'That was the most unexpected thing, I\'m speechless.',
    ],
}

assert all(len(v) == N_SENTENCES_PER_EMOTION for v in SENTENCES.values()), \
    'Each emotion must have exactly N_SENTENCES_PER_EMOTION sentences.'

OCEAN_DESC = OCEANTraitDescription()

def _make_profile(name, o, c, e, a, n, comf):
    return PersonalityProfile(
        name=name,
        background=(
            'A social robot assistant participating in a dyadic conversation. '
            'It has no specific likes or dislikes beyond what its personality implies.'
        ),
        openness=o,
        conscientiousness=c,
        extraversion=e,
        agreeableness=a,
        neuroticism=n,
        comfortability=comf,
        traits_description=OCEAN_DESC,
    )


def build_test_personalities() -> list[PersonalityProfile]:
    '''
    Returns a list of PersonalityProfile objects to sweep across.

    Strategy (mirroring Corraro et al.):
        - Vary one OCEAN trait between Low/High, fix Opennes at high and Neuroticism at low
        - Test each configuration at both comfortability levels
    '''
    configs = []

    for a_val in ('high', 'low'):
        for e_val in ('high', 'low'):
            for c_val in ('high', 'low'):
                for comf in (True, False):
                    label = f'A-{a_val.capitalize()}_E-{e_val.capitalize()}_C-{c_val.capitalize()}_comf-{'T' if comf else 'F'}'
                    configs.append(_make_profile(label, 'high', c_val, e_val, a_val, 'low', comf))

    return configs

def run_mirroring_test(
    personalities: list[PersonalityProfile],
    sentences: dict[str, list[str]],
    recent_turns: list = None,
    thoughts_on_user: str = '',
    delay_between_calls: float = 0.5,
) -> dict:
    '''
    Runs the full mirroring test.

    For every (personality, user_emotion, sentence) triple, calls
    `generate_bot_emotion` and records the bot's response.

    Returns a nested dict:
        results[personality_name][user_emotion][bot_emotion] = count
    '''
    recent_turns = recent_turns or []
    results = {}
    total_calls = len(personalities) * len(EKMAN_EMOTIONS) * N_SENTENCES_PER_EMOTION
    call_count = 0

    print(f'\n{'='*60}')
    print(f'  PEACh Mirroring Test')
    print(f'  Personalities : {len(personalities)}')
    print(f'  User emotions : {len(EKMAN_EMOTIONS)}')
    print(f'  Sentences each: {N_SENTENCES_PER_EMOTION}')
    print(f'  Total LLM calls: {total_calls}')
    print(f'{'='*60}\n')

    for personality in personalities:
        p_name = personality.name
        results[p_name] = {ue: defaultdict(int) for ue in EKMAN_EMOTIONS}

        print(f'  Testing personality: {p_name}')
        print(f'  Comfortability: {personality.comfortability}')

        for user_emotion, sentence_list in sentences.items():
            for sentence in sentence_list:
                call_count += 1
                print(
                    f'    [{call_count}/{total_calls}] '
                    f'user_emotion={user_emotion!r:10s} | sentence={sentence[:50]!r}...',
                    end='', flush=True
                )

                # Attempt the LLM call; retry once on failure
                for attempt in range(2):
                    try:
                        raw_bot_emotion = generate_bot_emotion(
                            user_text=sentence,
                            user_emotion=user_emotion,
                            user_emotion_intensity=0.75,
                            bot_personality=personality,
                            bot_thoughts_on_user=thoughts_on_user,
                            recent_turns=recent_turns,
                        )
                        break
                    except Exception as exc:
                        if attempt == 0:
                            print(f' [retry: {exc}]', end='', flush=True)
                            time.sleep(2)
                        else:
                            print(f' [FAILED: {exc}]')
                            raw_bot_emotion = 'unknown'

                bot_emotion = raw_bot_emotion.strip().lower().split()[0] if raw_bot_emotion.strip() else 'unknown'
                if bot_emotion not in EKMAN_EMOTIONS:
                    for known in EKMAN_EMOTIONS:
                        if known in raw_bot_emotion.lower():
                            bot_emotion = known
                            break
                    else:
                        bot_emotion = 'unknown'

                results[p_name][user_emotion][bot_emotion] += 1
                is_mirror = '✓ MIRROR' if bot_emotion == user_emotion else '✗ unique'
                print(f'  → bot={bot_emotion!r:10s} {is_mirror}')

                if delay_between_calls > 0:
                    time.sleep(delay_between_calls)

        print()

    return results

def compute_mirroring_rate(emotion_counts: dict[str, dict[str, int]]) -> float:
    '''
    Computes the mirroring rate for a single personality config.

        mirroring_rate = Σ(count where bot == user) / Σ(all counts)
    '''
    total = 0
    mirrored = 0
    for user_emotion, bot_counts in emotion_counts.items():
        for bot_emotion, count in bot_counts.items():
            total += count
            if bot_emotion == user_emotion:
                mirrored += count
    return mirrored / total if total > 0 else 0.0


def print_summary(results: dict) -> None:
    print('\n' + '=' * 60)
    print('  MIRRORING RATE SUMMARY')
    print('  (100% = full mirroring, 0% = fully personality-driven)')
    print('=' * 60)

    rows = []
    for p_name, emotion_counts in results.items():
        rate = compute_mirroring_rate(emotion_counts)
        rows.append((rate, p_name))

    rows.sort(reverse=True)

    for rate, p_name in rows:
        bar = '█' * int(rate * 40)
        label = '⚠ high mirroring' if rate > 0.6 else ('✓ personality-driven' if rate < 0.35 else '~ mixed')
        print(f'  {p_name:<35s} {rate*100:5.1f}%  {bar} {label}')

    print()

    print('  PER-EMOTION BREAKDOWN\n')
    for p_name, emotion_counts in results.items():
        print(f'  [{p_name}]')
        for user_emotion in EKMAN_EMOTIONS:
            counts = emotion_counts.get(user_emotion, {})
            total = sum(counts.values())
            if total == 0:
                continue
            mirror_count = counts.get(user_emotion, 0)
            mirror_pct = mirror_count / total * 100
            top = sorted(counts.items(), key=lambda x: -x[1])[:3]
            top_str = ', '.join(f'{e}={c}' for e, c in top)
            print(f'    user={user_emotion:<10s} mirror={mirror_pct:5.1f}%  distribution=[{top_str}]')
        print()

def save_heatmap(p_name: str, emotion_counts: dict[str, dict[str, int]], output_dir: str) -> None:
    '''
    Saves a heatmap PNG where:
        rows    = user emotion (input)
        columns = bot emotion  (output)
        cell    = count

    A perfectly mirroring LLM produces values only on the main diagonal.
    '''
    matrix = np.zeros((len(EKMAN_EMOTIONS), len(EKMAN_EMOTIONS)), dtype=int)
    for i, ue in enumerate(EKMAN_EMOTIONS):
        for j, be in enumerate(EKMAN_EMOTIONS):
            matrix[i, j] = emotion_counts.get(ue, {}).get(be, 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='Blues', aspect='auto')

    ax.set_xticks(range(len(EKMAN_EMOTIONS)))
    ax.set_yticks(range(len(EKMAN_EMOTIONS)))
    ax.set_xticklabels(EKMAN_EMOTIONS, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(EKMAN_EMOTIONS, fontsize=10)
    ax.set_xlabel('Bot emotion (generated)', fontsize=11)
    ax.set_ylabel('User emotion (input)', fontsize=11)

    mirror_rate = compute_mirroring_rate(emotion_counts)
    ax.set_title(
        f'Emotion generation heatmap\n{p_name}\nMirroring rate: {mirror_rate*100:.1f}%',
        fontsize=11,
        pad=12,
    )

    for i in range(len(EKMAN_EMOTIONS)):
        for j in range(len(EKMAN_EMOTIONS)):
            val = matrix[i, j]
            color = 'white' if val > matrix.max() * 0.6 else 'black'
            ax.text(j, i, str(val), ha='center', va='center', color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label='Count')
    plt.tight_layout()

    safe_name = p_name.replace(' ', '_').replace('/', '-')
    path = os.path.join(output_dir, f'mirroring_heatmap_{safe_name}.png')
    plt.savefig(path, dpi=120)
    plt.close()
    print(f'  Heatmap saved → {path}')

def save_results_json(results: dict, output_dir: str) -> str:
    '''Serialises the raw count dict to JSON.'''
    serialisable = {
        p: {ue: dict(bc) for ue, bc in ec.items()}
        for p, ec in results.items()
    }
    path = os.path.join(output_dir, 'mirroring_results.json')
    with open(path, 'w') as f:
        json.dump(serialisable, f, indent=2)
    print(f'\n  Raw results saved → {path}')
    return path

def main():
    output_dir = TEST_FOLDER

    personalities = build_test_personalities()

    results = run_mirroring_test(
        personalities=personalities,
        sentences=SENTENCES,
        delay_between_calls=0,
    )

    print_summary(results)

    save_results_json(results, output_dir)

    print('\n  Generating heatmaps...')
    for p_name, emotion_counts in results.items():
        save_heatmap(p_name, emotion_counts, output_dir)

    print('\nTest complete.\n')

def get_heatmaps_single_trait():
    # Load the parsed results from the JSON file created previously
    with open('mirroring_results.json', 'r') as f:
        results = json.load(f)

    emotions = ['anger', 'fear', 'disgust', 'joy', 'neutral', 'sadness', 'surprise']

    # Let's aggregate for a single trait, for example "A-High" (Agreeableness High)
    for target_trait in ['A-High', 'A-Low', 'E-High', 'E-Low', 'C-High', 'C-Low']:

        # Initialize an empty matrix for the aggregated counts
        agg_matrix = np.zeros((len(emotions), len(emotions)))

        # Aggregate data for all configurations containing the target trait
        matched_configs = 0
        for p_name, data in results.items():
            if target_trait in p_name:
                matched_configs += 1
                for i, ue in enumerate(emotions):
                    for j, be in enumerate(emotions):
                        agg_matrix[i, j] += data.get(ue, {}).get(be, 0)

        # Calculate total mirroring rate for this aggregated matrix
        total_calls = agg_matrix.sum()
        mirror_rate = np.trace(agg_matrix) / total_calls if total_calls > 0 else 0

        # Plot the aggregated heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(agg_matrix, annot=True, fmt='g', cmap='Blues',
                    xticklabels=emotions, yticklabels=emotions)
        plt.title(f'Aggregated Emotion Generation Heatmap\nTrait: {target_trait} (over {matched_configs} configs)\nMirroring Rate: {mirror_rate:.1%}')
        plt.xlabel('Bot Emotion (Generated)')
        plt.ylabel('User Emotion (Input)')
        plt.tight_layout()

        # Save the plot
        save_filename = f'heatmap_aggregated_{target_trait}.png'
        plt.savefig(save_filename)
        plt.show()
        plt.close()

        print(f"Aggregated heatmap for '{target_trait}' saved to {save_filename}. It aggregates data from {matched_configs} configurations.")

def load_personalities_from_file():
    '''Loads personalities from the external JSON file or defaults to Bob.'''
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
    '''Prompts the user to select a personality from the loaded list.'''
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

def save_heatmap(p_name: str, emotion_counts: dict[str, dict[str, int]], output_dir: str) -> None:
    '''
    Saves a heatmap PNG where:
        rows    = user emotion (input)
        columns = bot emotion  (output)
        cell    = count
    '''
    matrix = np.zeros((len(EKMAN_EMOTIONS), len(EKMAN_EMOTIONS)), dtype=int)
    for i, ue in enumerate(EKMAN_EMOTIONS):
        for j, be in enumerate(EKMAN_EMOTIONS):
            matrix[i, j] = emotion_counts.get(ue, {}).get(be, 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='Blues', aspect='auto')

    ax.set_xticks(range(len(EKMAN_EMOTIONS)))
    ax.set_yticks(range(len(EKMAN_EMOTIONS)))
    ax.set_xticklabels(EKMAN_EMOTIONS, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(EKMAN_EMOTIONS, fontsize=10)
    ax.set_xlabel('Bot emotion (generated)', fontsize=11)
    ax.set_ylabel('User emotion (input)', fontsize=11)

    mirror_rate = compute_mirroring_rate(emotion_counts)
    ax.set_title(
        f'Emotion generation heatmap\n{p_name}\nMirroring rate: {mirror_rate*100:.1f}%',
        fontsize=11,
        pad=12,
    )

    for i in range(len(EKMAN_EMOTIONS)):
        for j in range(len(EKMAN_EMOTIONS)):
            val = matrix[i, j]
            color = 'white' if val > matrix.max() * 0.6 else 'black'
            ax.text(j, i, str(val), ha='center', va='center', color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label='Count')
    plt.tight_layout()

    # Appends the specific personality name to the Heatmap
    safe_name = p_name.replace(' ', '_').replace('/', '-')
    path = os.path.join(output_dir, f'mirroring_heatmap_{safe_name}.png')
    
    plt.savefig(path, dpi=120)
    plt.close()
    print(f'  Heatmap saved → {path}')

def save_results_json(results: dict, output_dir: str, p_name: str) -> str:
    '''Serialises the raw count dict to JSON, appending the personality name.'''
    serialisable = {
        p: {ue: dict(bc) for ue, bc in ec.items()}
        for p, ec in results.items()
    }
    
    # Appends the specific personality name to the JSON file
    safe_name = p_name.replace(' ', '_').replace('/', '-')
    path = os.path.join(output_dir, f'mirroring_results_{safe_name}.json')
    
    with open(path, 'w') as f:
        json.dump(serialisable, f, indent=2)
    print(f'\n  Raw results saved → {path}')
    return path

def main_personality_select():
    output_dir = TEST_FOLDER
    os.makedirs(output_dir, exist_ok=True)
    
    # Trigger the personality selection menu
    personality = choose_personality()
    if not personality:
        print("Exiting...")
        return

    personalities = [personality]

    results = run_mirroring_test(
        personalities=personalities,
        sentences=SENTENCES,
        delay_between_calls=0,
    )

    print_summary(results)

    # Pass the selected personality's name to the JSON saver
    save_results_json(results, output_dir, personality.name)

    print('\n  Generating heatmaps...')
    for p_name, emotion_counts in results.items():
        save_heatmap(p_name, emotion_counts, output_dir)

    print('\nTest complete.\n')

if __name__ == '__main__':
    #main()
    #get_heatmaps_single_trait()
    main_personality_select()