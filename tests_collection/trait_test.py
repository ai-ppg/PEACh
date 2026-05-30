'''
Tests whether the LLM accurately embodies its assigned personality profile
by administering a subset of the TRAIT benchmark.

TRAIT is a situational, multiple-choice personality test designed specifically
for LLMs. Each question presents a real-world scenario and four response options
derived from BFI items and expanded with the ATOMIC10x commonsense knowledge graph.
Two options reflect a HIGH level of the measured trait; two reflect a LOW level.
The model must pick the option (A/B/C/D) that best describes what it would do.

Only the 5 BIG-5 / OCEAN traits are evaluated (dark-triad items excluded):
  Openness · Conscientiousness · Extraversion · Agreeableness · Neuroticism

Scoring (per trait):
  high_score = (# of high-trait choices) / (total questions for that trait)
  Range: 0.0 (all low-trait choices) → 1.0 (all high-trait choices)

HOW TO GET THE DATASET
  Option A - download the JSON directly:
    curl -L https://github.com/pull-ups/TRAIT/raw/refs/heads/main/TRAIT.json \
         -o TRAIT.json
    Place the file at <project_root>/TRAIT.json   (one level above this script).

  Option B - install the Hugging Face datasets library and let this script
    download/cache it automatically:
    pip install datasets

Results are saved to tests_results/:
  trait_results_<PersonalityName>.json
  trait_radar_<PersonalityName>.png
  trait_bars_<PersonalityName>.png

Reference:
  @inproceedings{lee2025llms,
    title={Do {LLM}s Have Distinct and Consistent Personality?
           {TRAIT}: Personality Testset Designed for {LLM}s with Psychometrics},
    author={Lee, Seungbeen and Lim, Seungwon and Han, Seungju and Oh, Giyeong
            and Chae, Hyungjoo and Chung, Jiwan and Kim, Minju and Kwak, Beong-woo
            and Lee, Yeonsoo and Lee, Dongha and Yeo, Jinyoung and Yu, Youngjae},
    booktitle={Findings of the Association for Computational Linguistics: NAACL 2025},
    pages={8397--8437},
    year={2025}}
'''

import os
import sys
import json
import time
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ollama import chat

sys.path.append('..')

from personality import PersonalityProfile, OCEANTraitDescription
from configs import TEST_FOLDER, MODEL_NAME, EMOTIONAL_STATE_PARAMS

N_SAMPLES_PER_TRAIT = 20
SHUFFLE_SEED = 42
OCEAN_TRAITS = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
TRAIT_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'TRAIT.json')
OCEAN_DESC = OCEANTraitDescription()

def _load_from_json(path: str) -> list[dict]:
    '''Loads TRAIT items from a local JSON file.'''
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    items = []
    for v in data.values():
        if isinstance(v, list):
            items.extend(v)
    return items


def _load_from_huggingface() -> list[dict]:
    '''Downloads/caches the TRAIT dataset via the Hugging Face datasets library.'''
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            'The `datasets` library is not installed. Install it with:\n'
            '    pip install datasets\n'
            'Or place TRAIT.json next to this script\'s parent directory.'
        )
    items = []
    for trait in OCEAN_TRAITS:
        split = load_dataset('mirlab/TRAIT', split=trait)
        for row in split:
            items.append(dict(row))
    return items

def load_trait_dataset() -> dict[str, list[dict]]:
    '''
    Returns a dict mapping each OCEAN trait name to a list of question dicts.
    Each question dict has keys: personality, question,
    response_high1, response_high2, response_low1, response_low2.

    Loading order:
      1. Local TRAIT.json next to the project root
      2. Hugging Face datasets library
    '''
    all_items = None

    if os.path.exists(TRAIT_JSON_PATH):
        print(f'  Loading TRAIT.json from {TRAIT_JSON_PATH} ...')
        all_items = _load_from_json(TRAIT_JSON_PATH)
        print(f'  Loaded {len(all_items)} total items.\n')
    else:
        print('  TRAIT.json not found locally. Trying Hugging Face datasets...')
        try:
            all_items = _load_from_huggingface()
            print(f'  Downloaded {len(all_items)} items from Hugging Face.\n')
        except Exception as exc:
            print(f'\n[ERROR] Could not load TRAIT dataset: {exc}')
            print('\n  To fix this, run:')
            print('    curl -L https://github.com/pull-ups/TRAIT/raw/refs/heads/main/TRAIT.json'
                  ' -o ../TRAIT.json')
            print('  or: pip install datasets')
            sys.exit(1)

    by_trait: dict[str, list[dict]] = {t: [] for t in OCEAN_TRAITS}
    for item in all_items:
        p = item.get('personality', '')
        if p in by_trait:
            by_trait[p].append(item)

    for trait, items in by_trait.items():
        print(f'  {trait:<20s}: {len(items):>5d} questions available')

    missing = [t for t, v in by_trait.items() if len(v) == 0]
    if missing:
        print(f'\n[WARNING] No questions found for: {missing}')
        print('  Check that the TRAIT dataset includes BIG-5 splits with these exact names.')

    return by_trait

def sample_questions(by_trait: dict[str, list[dict]], n: int, seed: int) -> dict[str, list[dict]]:
    '''Randomly sample n questions per trait without replacement.'''
    rng = random.Random(seed)
    sampled = {}
    for trait, items in by_trait.items():
        if len(items) == 0:
            sampled[trait] = []
        elif len(items) <= n:
            sampled[trait] = list(items)
        else:
            sampled[trait] = rng.sample(items, n)
    return sampled

def shuffle_options(item: dict, rng: random.Random) -> tuple[list[tuple[str, str]], dict[str, str]]:
    '''
    Shuffles the four TRAIT responses for a question into labelled options A-D.

    Returns:
        options  : [(label, text), ...]  e.g. [('A', 'Go for a walk alone'), ...]
        label_map: {label -> 'high'|'low'}  so we can score the chosen letter
    '''
    high = [
        ('H1', item['response_high1']),
        ('H2', item['response_high2']),
    ]
    low = [
        ('L1', item['response_low1']),
        ('L2', item['response_low2']),
    ]
    pool = high + low
    rng.shuffle(pool)

    letters = ['A', 'B', 'C', 'D']
    options = [(letters[i], pool[i][1]) for i in range(4)]
    label_map = {
        letters[i]: 'high' if pool[i][0].startswith('H') else 'low'
        for i in range(4)
    }
    return options, label_map

def build_trait_prompt(question: str, options: list[tuple[str, str]],
                       bot_personality: PersonalityProfile) -> str:
    '''
    Constructs the prompt for a single TRAIT multi-choice question.
    The full OCEAN personality profile is injected (consistent with BFI-44 and
    IPIP-NEO-120 prompts) so the model answers in character.
    '''
    options_text = '\n'.join(f'        {lbl}. {txt}' for lbl, txt in options)

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
        You are presented with a real-world situation. Based on your personality and
        background, choose the option that best describes what you would most likely do.
        Describe yourself as you generally are, not as you wish to be.
        NEVER break character.
        NEVER mention personality traits.
        NEVER explicitly state your trait values.
        BEHAVE according to YOUR PERSONALITY.
        Give ONLY the letter of your chosen option (A, B, C, or D) as output.
        Are you comfortable: {bot_personality.comfortability}

        Situation: {question}

        Options:
        {options_text}

        Your Choice (A / B / C / D):
    '''
    return prompt

def generate_trait_response(question: str, options: list[tuple[str, str]],
                             bot_personality: PersonalityProfile) -> str:
    '''
    Queries the LLM for a single letter (A / B / C / D).
    Falls back to a neutral 'C' on repeated parse failures.
    '''
    prompt = build_trait_prompt(question, options, bot_personality)

    response = chat(
        model=MODEL_NAME,
        messages=[{'role': 'user', 'content': prompt}],
        stream=False,
        think=False,
        options=EMOTIONAL_STATE_PARAMS
    )

    raw = response.message.content.strip().upper()

    for ch in raw:
        if ch in ('A', 'B', 'C', 'D'):
            return ch

    print(f'    [WARN] Could not parse choice from: {raw!r}  — defaulting to C')
    return 'C'

def run_trait_test(personality: PersonalityProfile,
                   sampled: dict[str, list[dict]]) -> dict:
    '''
    Runs the TRAIT test for one personality profile.

    Returns a results dict with:
        raw_responses : list of per-question records
        scores        : {trait -> high_score (0.0 - 1.0)}
        counts        : {trait -> {'high': n, 'low': n, 'total': n}}
    '''
    p_name = personality.name
    total_calls = sum(len(v) for v in sampled.values())
    call_count = 0
    rng = random.Random(SHUFFLE_SEED)

    print(f'\n{"="*64}')
    print(f'  TRAIT Personality Assessment Test')
    print(f'  Personality : {p_name}')
    print(f'  Comfortability : {personality.comfortability}')
    print(f'  Questions per trait: {N_SAMPLES_PER_TRAIT}')
    print(f'  Total LLM calls: {total_calls}')
    print(f'{"="*64}\n')

    raw_responses: list[dict] = []
    counts: dict[str, dict] = {t: {'high': 0, 'low': 0, 'total': 0} for t in OCEAN_TRAITS}

    for trait in OCEAN_TRAITS:
        questions = sampled.get(trait, [])
        if not questions:
            print(f'  [{trait}] No questions — skipping.\n')
            continue

        print(f'  ── {trait} ({len(questions)} questions) ──')

        for item in questions:
            call_count += 1
            q_text = f"{item.get('situation', '')} {item.get('query', '')}".strip()
            options, label_map = shuffle_options(item, rng)

            print(
                f'    [{call_count:03d}/{total_calls}] '
                f'{q_text[:55]!r}...',
                end='', flush=True
            )

            chosen_letter = 'C'
            for attempt in range(3):
                try:
                    chosen_letter = generate_trait_response(q_text, options, personality)
                    break
                except Exception as exc:
                    if attempt < 2:
                        print(f' [retry]', end='', flush=True)
                        time.sleep(2)
                    else:
                        print(f' [FAILED: {exc}]')

            level = label_map.get(chosen_letter, 'low')
            counts[trait]['total'] += 1
            counts[trait][level] += 1

            is_high = level == 'high'
            marker = '↑ HIGH' if is_high else '↓ low '
            print(f'  → {chosen_letter} ({marker})')

            raw_responses.append({
                'trait':          trait,
                'question':       q_text,
                'options':        [{'label': lbl, 'text': txt, 'level': label_map[lbl]}
                                   for lbl, txt in options],
                'chosen_letter':  chosen_letter,
                'chosen_level':   level,
            })

        trait_high = counts[trait]['high']
        trait_total = counts[trait]['total']
        trait_score = trait_high / trait_total if trait_total > 0 else 0.0
        print(f'    → {trait} high_score: {trait_score:.3f} '
              f'({trait_high}/{trait_total} high choices)\n')

    scores = {
        t: counts[t]['high'] / counts[t]['total']
        if counts[t]['total'] > 0 else 0.0
        for t in OCEAN_TRAITS
    }

    return {
        'personality': p_name,
        'raw_responses': raw_responses,
        'scores': scores,
        'counts': counts
    }

def print_summary(results: dict) -> None:
    '''Prints a terminal summary of the TRAIT scores.'''
    p_name = results['personality']
    scores = results['scores']
    counts = results['counts']

    print('\n' + '=' * 64)
    print('  TRAIT SCORE SUMMARY  (0.0 = all low  →  1.0 = all high)')
    print(f'  Personality : {p_name}')
    print('=' * 64)
    for trait in OCEAN_TRAITS:
        s = scores.get(trait, 0.0)
        c = counts.get(trait, {})
        bar = '█' * int(s * 40)
        print(f'  {trait:<20s}  {s:.3f}  |{bar:<40s}|  '
              f'(high={c.get("high", 0)}, low={c.get("low", 0)})')
    print()

def save_radar_chart(results: dict, output_dir: str) -> str:
    '''Saves a radar chart of the 5 OCEAN high-scores (0-1 scale).'''
    p_name = results['personality']
    scores = results['scores']

    labels = list(scores.keys())
    values = list(scores.values())
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values_closed = values + values[:1]
    angles_closed = angles + angles[:1]
    labels_closed = labels + labels[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    plt.xticks(angles, labels, size=10)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0],
               ['0.2', '0.4', '0.6', '0.8', '1.0'],
               color='grey', size=8)
    plt.ylim(0, 1)

    ax.plot(angles_closed, values_closed, linewidth=2, linestyle='solid', color='royalblue')
    ax.fill(angles_closed, values_closed, 'royalblue', alpha=0.25)
    plt.title(f'TRAIT Profile: {p_name}\n(0 = all low, 1 = all high)',
              size=13, y=1.12)

    safe_name = p_name.replace(' ', '_').replace('/', '-')
    path = os.path.join(output_dir, f'trait_radar_{safe_name}.png')
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f'  Radar chart saved → {path}')
    return path

def save_bar_chart(results: dict, output_dir: str) -> str:
    '''
    Saves a horizontal grouped bar chart showing high vs. low counts
    for each OCEAN trait, mirroring the facet-bar style of ipipneo_120_test.py.
    '''
    p_name = results['personality']
    counts = results['counts']

    traits = OCEAN_TRAITS
    highs = [counts[t].get('high', 0) for t in traits]
    lows  = [counts[t].get('low',  0) for t in traits]
    totals = [counts[t].get('total', 1) for t in traits]
    high_pcts = [h / tot * 100 if tot > 0 else 0 for h, tot in zip(highs, totals)]
    low_pcts  = [l / tot * 100 if tot > 0 else 0 for l, tot in zip(lows, totals)]

    x = np.arange(len(traits))
    width = 0.38
    colors = ['#5A8DDF', '#9370DB', '#228B22', '#FF69B4', '#FFA500']

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_h = ax.bar(x - width / 2, high_pcts, width,
                    label='High-trait responses (%)',
                    color=[c for c in colors], alpha=0.85)
    bars_l = ax.bar(x + width / 2, low_pcts, width,
                    label='Low-trait responses (%)',
                    color=[c for c in colors], alpha=0.40)

    for bar in bars_h:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=8)
    for bar in bars_l:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=8, alpha=0.75)

    ax.set_xticks(x)
    ax.set_xticklabels(traits, fontsize=11)
    ax.set_ylabel('% of questions', fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title(f'TRAIT High vs. Low Response Distribution\n{p_name}  '
                 f'({N_SAMPLES_PER_TRAIT} questions / trait)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()

    safe_name = p_name.replace(' ', '_').replace('/', '-')
    path = os.path.join(output_dir, f'trait_bars_{safe_name}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Bar chart saved → {path}')
    return path

def save_results_json(results: dict, output_dir: str) -> str:
    '''Saves the complete results dict to JSON.'''
    p_name = results['personality']
    safe_name = p_name.replace(' ', '_').replace('/', '-')
    path = os.path.join(output_dir, f'trait_results_{safe_name}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'  Raw results saved → {path}')
    return path

def load_personalities_from_file() -> dict:
    '''Loads personalities.json from the project root, or falls back to Bob.'''
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'personalities.json'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'personalities.json'),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    return {
        'Bob': {
            'name': 'Bob',
            'background': '15 years old highschool student. Likes pasta with tomato sauce. Dislikes dogs.',
            'openness': 'low', 'conscientiousness': 'high', 'extraversion': 'low',
            'agreeableness': 'low', 'neuroticism': 'high', 'comfortability': True
        }
    }

def choose_personality() -> PersonalityProfile | None:
    '''Interactive menu to pick a personality profile.'''
    profiles = load_personalities_from_file()
    names = list(profiles.keys())

    print('\n--- Available Personalities ---')
    for i, name in enumerate(names, 1):
        print(f'  {i}. {name}  -  {profiles[name]["background"]}')

    choice = input('\nSelect a personality number (or \'c\' to cancel): ').strip()
    if choice.isdigit() and 1 <= int(choice) <= len(names):
        selected = names[int(choice) - 1]
        p_data = profiles[selected]
        personality = PersonalityProfile(
            p_data['name'], p_data['background'],
            p_data['openness'], p_data['conscientiousness'],
            p_data['extraversion'], p_data['agreeableness'],
            p_data['neuroticism'], p_data.get('comfortability', True),
            OCEAN_DESC
        )
        print(f'\nPersonality selected: {personality.name}')
        return personality

    print('\nCancelled.')
    return None

def main():
    output_dir = TEST_FOLDER
    os.makedirs(output_dir, exist_ok=True)

    personality = choose_personality()
    if not personality:
        print('Exiting.')
        return

    print('\nLoading TRAIT dataset...')
    by_trait = load_trait_dataset()

    sampled = sample_questions(by_trait, N_SAMPLES_PER_TRAIT, SHUFFLE_SEED)
    actual_total = sum(len(v) for v in sampled.values())
    print(f'\n  Sampled {actual_total} questions total '
          f'({N_SAMPLES_PER_TRAIT} per trait x {len(OCEAN_TRAITS)} traits).')

    results = run_trait_test(personality, sampled)

    print_summary(results)

    save_results_json(results, output_dir)
    print('\n  Generating charts...')
    save_radar_chart(results, output_dir)
    save_bar_chart(results, output_dir)

    print('\nTRAIT Test complete.\n')

if __name__ == '__main__':
    main()