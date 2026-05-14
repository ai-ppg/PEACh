# PEACh - Personality-driven Emotional Agent for Chatting

PEACh is a chatbot that simulates a character with a persistent personality defined by the Big Five (OCEAN) model. It recognises the user's emotions, maintains its own internal emotional state, reflects on the conversation periodically, and builds a long-term memory of past interactions.

---

## Table of Contents

1. [Architecture overview](#architecture-overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
   - [Python 3.12](#1-python-312)
   - [Ollama](#2-ollama)
   - [Pull the required models](#3-pull-the-required-models)
   - [Clone the project and install Python dependencies](#4-clone-the-project-and-install-python-dependencies)
   - [Download the EmoBART model](#5-download-the-emobart-model)
4. [Project structure](#project-structure)
5. [Configuration](#configuration)
6. [Running PEACh](#running-peach)
7. [Menu walkthrough](#menu-walkthrough)
8. [Personality system](#personality-system)
9. [Memory system](#memory-system)
10. [Tests](#tests)
11. [Troubleshooting](#troubleshooting)

---

## Architecture overview

```
User input
    │
    ▼
EmoBART ──► user emotion + intensity
    │
    ▼
Qwen3.5-4B (Ollama)
    ├── emotion generator   ──► bot internal emotion
    ├── response generator  ──► bot reply  ◄── long-term memory (ChromaDB + embeddinggemma) + working memory (last 3 turns + current reflection)
    ├── reflection          ──► updated thoughts on user   (every 3 turns)
    └── comfortability      ──► updated comfort flag       (every 3 turns)
```

There are three layers of memory:

| Layer | Scope | Storage |
|---|---|---|
| Working memory | Last 3 turns | In-process Python list |
| Reflection | Periodic summary of feelings and insights | In-process string, persisted in save file |
| Long-term memory | Full conversation history | ChromaDB on disk, one folder per session |

---

## Requirements

| Component | Version |
|---|---|
| Python | 3.12 |
| Ollama | latest |
| LLM (chat) | `qwen3.5:4b` |
| Embeddings | `embeddinggemma` |
| Emotion recogniser | `lzw1008/Emobart-large` (downloaded automatically from Hugging Face) |

A CUDA-capable GPU is strongly recommended. The project runs on CPU but generation will be significantly slower.

---

## Installation

### 1. Python 3.12

**macOS / Linux — via pyenv (recommended)**

```bash
# Install pyenv if you don't have it
curl https://pyenv.run | bash

# Reload your shell, then:
pyenv install 3.12
pyenv local 3.12        # sets 3.12 for this directory only
python --version        # should print Python 3.12.x
```

**Windows**

Download the Python 3.12 installer from https://www.python.org/downloads/ and make sure to tick *Add Python to PATH* during setup.

---

### 2. Ollama

Ollama is the local model runtime that PEACh uses for the chat LLM and the embedding model.

**macOS**

```bash
brew install ollama
```

**Linux**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows**

Download the installer from https://ollama.com/download and run it.

After installation, start the Ollama daemon (it may start automatically as a service on Windows/macOS):

```bash
ollama serve
```

Leave this terminal open. All subsequent steps assume the daemon is running.

---

### 3. Pull the required models

Open a **new terminal** (keep `ollama serve` running in the first one) and pull both models:

```bash
# Chat / generation model
ollama pull qwen3.5:4b

# Embedding model
ollama pull embeddinggemma
```

Verify they are available:

```bash
ollama list
```

You should see both `qwen3.5:4b` and `embeddinggemma` in the output.

> **Changing models** — to use different Ollama models, update `MODEL_NAME` and `EMBEDDING_MODEL_NAME` in `configs.py` to match the names shown by `ollama list`, and tune the generation parameters accordingly.

---

### 4. Clone the project and install Python dependencies

```bash
git clone <your-repo-url>
cd peach

# Create a virtual environment with Python 3.12
python3.12 -m venv .venv

# Activate it — macOS / Linux
source .venv/bin/activate

# Activate it — Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Install all Python dependencies
pip install -r requirements.txt
```

**GPU acceleration (optional but recommended)**

If you have a CUDA GPU, install the matching PyTorch build *before* the rest of the requirements so it is not overwritten:

```bash
# Example for CUDA 12.1 — check https://pytorch.org/get-started for your CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

### 5. Download the EmoBART model

EmoBART is downloaded automatically from Hugging Face the first time you run PEACh. To pre-download it manually (useful on machines that will later run offline):

```bash
python emotion_recognizer_bart.py
```

The model (~1.6 GB) is saved to `./local_emobart_model/` and will not be re-downloaded on subsequent runs.

---

## Project structure

```
peach/
├── main.py                       # Entry point — run this
├── menu.py                       # CLI menu and session management
├── bot.py                        # PEACh class — chat loop and state
├── personality.py                # PersonalityProfile dataclass and OCEAN descriptions
├── emotion_recognizer_bart.py    # EmoBART-based user emotion recognition
├── emotion_generator.py          # Bot internal emotion + comfortability (Ollama)
├── response_generator.py         # Bot reply generation (Ollama)
├── working_memory.py             # Rolling turn buffer and periodic reflection
├── long_term_memory.py           # ChromaDB-backed long-term memory
├── configs.py                    # All tuneable parameters in one place
├── personalities.json            # Personality profiles (edited via the menu)
├── requirements.txt
│
├── local_emobart_model/          # EmoBART weights (auto-created on first run)
├── long_memory/                  # ChromaDB stores — one subfolder per chat session
│   └── bob_a3f9c1d2/
├── chat_saves/                   # JSON save files — one per chat session
│   └── bob_a3f9c1d2.json
└── tests_collection/             # Evaluation scripts
    ├── mirroring_test.py
    ├── bfi_44_test.py
    ├── ipipneo_120_test.py
    └── tests_results/
```

---

## Configuration

All parameters live in `configs.py`:

```python
MODEL_NAME           = 'qwen3.5:4b'      # Ollama chat model
EMBEDDING_MODEL_NAME = 'embeddinggemma'  # Ollama embedding model
EMOBART_MODEL_ID     = 'lzw1008/Emobart-large'
SAVE_EMOBART_DIR     = './local_emobart_model'
CHROMADB_COLLECTION  = './long_memory'
CHAT_SAVES_DIR       = './chat_saves'
TEST_FOLDER          = './tests_results'

WM_BUFFER_SIZE       = 3     # Working-memory window AND reflection frequency (turns)

LTM_K                = 4     # Memories injected into each response prompt
LTM_RECENCY_ALPHA    = 0.3   # 0 = pure semantic retrieval, 1 = pure recency
LTM_RECENCY_DECAY    = 0.05  # Exponential decay rate per turn for recency scoring
```

Generation parameters (temperature, top-p, top-k, presence penalty) are separately configurable for the response, reflection, and emotional-state generation tasks.

---

## Running PEACh

Make sure `ollama serve` is running, then:

```bash
python main.py
```

On the very first run, EmoBART will be downloaded from Hugging Face. This happens once and takes a few minutes depending on your connection.

---

## Menu walkthrough

```
==============================
       PEACh Main Menu
       Active Bot: Bob
==============================
1. Start New Chat
2. Choose Personality
3. Create Personality
4. Delete Personality
5. Save Chat
6. Load Chat
7. Delete Saved Chat
8. Exit
```

The active personality is shown in the menu header. All operations act on that personality unless noted otherwise.

---

### 1. Start New Chat

Starts a conversation with the currently active personality. Type your message and press Enter. Type `q` to quit back to the main menu.

Each session gets a unique ID (e.g. `bob_a3f9c1d2`) that ties together the save file and the long-term memory store. Debug lines printed during the conversation show the full emotion pipeline:

```
-------------------- TURN 1 --------------------

You: hey, I made pasta today!
[Debug] User Emotion: joy, Intensity: 0.81
[Debug] Bot Emotion: neutral
[LTM] Retrieved 0 memories.
Bob: Oh, pasta again? Sure, whatever.
```

---

### 2. Choose Personality

Lists all available personalities and lets you switch the active one. The change takes effect on the next *Start New Chat* — it does not affect a session that is already running.

---

### 3. Create Personality

Interactive wizard to define a new character:

- **Name** — the bot's display name.
- **Background** — free-text description of the character's story, speech style, likes, dislikes, and quirks. The richer this is, the more consistent the character's behaviour will be.
- **OCEAN traits** — enter `high` or `low` for each of the five dimensions (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism).
- **Initial comfortability** — whether the bot starts comfortable with the user (`y`) or not (`n`).

The profile is saved to `personalities.json` and immediately available for selection.

---

### 4. Delete Personality

Lists all personalities and lets you remove one. The built-in `Bob` profile cannot be deleted (it is the fallback default). If you delete the currently active personality, the bot automatically reverts to Bob.

---

### 5. Save Chat

Saves the current session's lightweight state to `chat_saves/<chat_id>.json`:

- Turn number
- Last 3 working-memory turns
- Reflection text (thoughts on the user)
- Comfortability flag
- Chat ID (links back to the ChromaDB store)

The long-term memory (ChromaDB) is written to disk after every turn and does not need an explicit save.

Re-saving an ongoing session overwrites only its own file. Different sessions are never overwritten.

---

### 6. Load Chat

Lists all saves in `chat_saves/`, sorted by most-recent turn first:

```
--- Saved Chats ---
  1. [Bob]    turn 24  –  id: bob_a3f9c1d2
  2. [Aria]   turn 9   –  id: aria_3c71e804
```

Selecting a save restores the full state and reopens the matching ChromaDB store, so long-term memories from the previous session are immediately available.

---

### 7. Delete Saved Chat

Same picker as Load. Selecting a save permanently deletes both the JSON file and the ChromaDB folder for that session. This action cannot be undone.

> **Note:** if the session you want to delete is still loaded in memory (i.e. you saved and then tried to delete without restarting), exit and re-run the script first, then delete.

---

## Personality system

Each personality is composed of two parts:

**Background** — a plain-English description injected directly into every LLM prompt. Include age, occupation, speech style, likes, dislikes, and any other traits you want consistently reflected.

**OCEAN traits** — each dimension maps to a detailed natural-language description that is also injected into the prompt:

| Trait | Low | High |
|---|---|---|
| Openness | Practical, conventional, avoids creative ideas | Imaginative, curious, appreciates art and new experiences |
| Conscientiousness | Impulsive, messy, avoids difficult tasks | Organised, self-disciplined, goal-driven |
| Extraversion | Introverted, prefers solo activities, reserved | Sociable, assertive, energetic, talkative |
| Agreeableness | Selfish, distrustful, holds grudges | Empathetic, cooperative, forgiving |
| Neuroticism | Calm, emotionally stable, patient | Anxious, emotionally reactive, tense |

**Comfortability** is a boolean flag updated every `WM_BUFFER_SIZE` turns. It reflects whether the bot currently feels at ease with this specific user and influences both emotional responses and the tone of replies.

---

## Memory system

### Working memory

A rolling window of the last `WM_BUFFER_SIZE` turns (default: 3). Each entry stores the user text, user emotion and intensity, bot emotion, and bot reply. This buffer is passed verbatim into every prompt.

### Reflection

Every `WM_BUFFER_SIZE` turns, the LLM generates a short internal monologue (4–5 sentences) summarising the bot's feelings and insights about the ongoing conversation. This replaces the previous reflection and is persisted in the save file.

### Long-term memory

Each session has a dedicated ChromaDB persistent store under `./long_memory/<chat_id>/`. Two types of entries are stored:

- **Turn memories** — a one-line summary of each turn, written immediately after the bot replies.
- **Reflection memories** — the full reflection text, written every `WM_BUFFER_SIZE` turns.

Before each response, the `LTM_K` most relevant memories are retrieved using a combined semantic + recency score:

```
combined = (1 - alpha) * semantic_score + alpha * recency_score
recency  = exp(-decay * (current_turn - memory_turn))
```

With the default `LTM_RECENCY_ALPHA = 0.3`, semantic relevance dominates but recent turns get a moderate boost. Increase alpha toward `1.0` to bias strongly toward recency; decrease toward `0.0` for purely semantic retrieval.

---

## Tests

Evaluation scripts are located in `tests_collection/` and must be run from inside that directory (or with the project root on `sys.path`).

### Mirroring test (`mirroring_test.py`)

Tests whether the emotion generator produces personality-driven emotional responses or simply mirrors the user's emotion. For each of the 7 basic Ekman emotions, a set of sentences is fed to `generate_bot_emotion` across multiple personality configurations (sweeping Agreeableness, Extraversion, and Conscientiousness at both comfortability levels). Results are saved as:

- `tests_results/mirroring_results.json` — raw counts per (personality, user emotion, bot emotion).
- `tests_results/mirroring_heatmap_<personality>.png` — one heatmap per personality configuration.

Run with:

```bash
cd tests_collection
python mirroring_test.py
```

To regenerate only the aggregated per-trait heatmaps from a previously saved JSON:

```python
# In mirroring_test.py, uncomment:
get_heatmaps_single_trait()
```

### BFI-44 test (`bfi_44_test.py`)

Administers the 44-item Big Five Inventory to the bot to verify that its self-reported traits align with the configured OCEAN profile.

### IPIP-NEO-120 test (`ipipneo_120_test.py`)

Administers the 120-item IPIP-NEO questionnaire to the bot for a more detailed personality validation.

Both questionnaire tests write their results to `tests_results/`.

---

## Troubleshooting

**`ollama: command not found`**
Ollama is not on your PATH. On Linux, add `/usr/local/bin` to your PATH or restart your shell after installation.

**`Connection refused` when calling Ollama**
The daemon is not running. Start it with `ollama serve` in a separate terminal and leave it open.

**EmoBART download fails**
`lzw1008/Emobart-large` is a public model — no Hugging Face login is required. Check your internet connection and try again. If you are behind a proxy, set the `HTTPS_PROXY` environment variable before running.

**CUDA out of memory**
Reduce the model size or set `device_map='cpu'` in `emotion_recognizer_bart.py`. Alternatively, run EmoBART on CPU and let Ollama handle GPU allocation for the chat model.

**Bot responses contain raw prompt text or reasoning tokens**
This happens when `think=False` is not supported by an older version of Ollama. Update Ollama to the latest release:
```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew upgrade ollama
```

**`ValueError: Openness must be in ['High', 'Low']`**
A `personalities.json` entry has a trait value other than `"high"` or `"low"`. Open the file and correct the offending value.

**Deleting a chat fails with "still in memory" warning**
Exit PEACh completely, re-run `python main.py`, and then use option 7 to delete the save.