# PEACh — Personality-Driven Emotionally Aware Chatbot

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
   - [Start a new chat](#1-start-new-chat)
   - [Choose personality](#2-choose-personality)
   - [Create personality](#3-create-personality)
   - [Delete personality](#4-delete-personality)
   - [Save chat](#5-save-chat)
   - [Load chat](#6-load-chat)
   - [Delete saved chat](#7-delete-saved-chat)
8. [Personality system](#personality-system)
9. [Memory system](#memory-system)
10. [Troubleshooting](#troubleshooting)

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
    ├── response generator  ──► bot reply  ◄── long-term memory (ChromaDB, embeddinggemma)
    ├── reflection          ──► updated thoughts on user   (every 3 turns)
    └── comfortability      ──► updated comfort flag       (every 3 turns)
```

There are three layers of memory:

| Layer | Scope | Storage |
|---|---|---|
| Working memory | Last 3 turns | In-process Python list |
| Reflection | Periodic summary of feelings/insights | In-process string, persisted in save file |
| Long-term memory | Full conversation history | ChromaDB on disk, per-session folder |

---

## Requirements

| Component | Version |
|---|---|
| Python | 3.12 |
| Ollama | latest |
| LLM (chat) | `qwen3.5:4b` |
| Embeddings | `embeddinggemma` (any Ollama embedding model) |
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

After installation, start the Ollama daemon (it may start automatically as a service):

```bash
ollama serve
```

---

### 3. Pull the required models

Open a new terminal (leave `ollama serve` running in the background) and pull both models:

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

> **Changing models** — if you want to use different models, update `MODEL_NAME` and `EMBEDDING_MODEL_NAME` in `configs.py` to match the names shown by `ollama list`.

---

### 4. Clone the project and install Python dependencies

```bash
git clone <your-repo-url>
cd peach

# Create and activate a virtual environment
python3.12 -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

PyTorch is included via the `torch` entry in `requirements.txt`. If you have a CUDA GPU and want GPU acceleration, install the matching CUDA build of PyTorch first (before the rest of the requirements):

```bash
# Example for CUDA 12.1 — check https://pytorch.org/get-started for your version
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

### 5. Download the EmoBART model

EmoBART is downloaded automatically from Hugging Face the first time you run the project. To pre-download it manually (useful on a machine without internet access later):

```bash
python emotion_recognizer_bart.py
```

The model is saved to `./local_emobart_model/` and will not be re-downloaded on subsequent runs.

---

## Project structure

```
peach/
├── main.py                   # Entry point
├── menu.py                   # CLI menu and session management
├── bot.py                    # PEACh class — chat loop and state
├── personality.py            # PersonalityProfile and OCEAN descriptions
├── emotion_recognizer_bart.py# EmoBART-based user emotion recognition
├── emotion_generator.py      # Bot internal emotion + comfortability (Ollama)
├── response_generator.py     # Bot reply generation (Ollama)
├── working_memory.py         # Rolling turn buffer and reflection generation
├── long_term_memory.py       # ChromaDB-backed long-term memory
├── configs.py                # All tuneable parameters
├── requirements.txt
│
├── local_emobart_model/      # EmoBART weights (auto-created on first run)
├── long_memory/              # ChromaDB stores, one subfolder per session
│   └── bob_a3f9c1d2/         # Example: Bob persona, session id a3f9c1d2
├── chat_saves/               # JSON save files, one per session
│   └── bob_a3f9c1d2.json
└── personalities.json        # User-created personality profiles
```

---

## Configuration

All parameters are in `configs.py`:

```python
MODEL_NAME           = 'qwen3.5:4b'      # Ollama chat model
EMBEDDING_MODEL_NAME = 'embeddinggemma'  # Ollama embedding model
EMOBART_MODEL_ID     = 'lzw1008/Emobart-large'
SAVE_EMOBART_DIR     = './local_emobart_model'
CHROMADB_COLLECTION  = './long_memory'

WM_BUFFER_SIZE       = 3    # Working memory window (turns)
                            # Also controls how often reflection runs

LTM_K                = 4    # Memories injected into each prompt
LTM_RECENCY_ALPHA    = 0.3  # 0 = pure semantic, 1 = pure recency
LTM_RECENCY_DECAY    = 0.05 # Decay rate per turn for recency scoring
```

---

## Running PEACh

Make sure the Ollama daemon is running, then:

```bash
python main.py
```

On the very first run, EmoBART will be downloaded from Hugging Face (~1.6 GB). This happens once.

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

### 1. Start New Chat

Starts a conversation with the currently active personality (shown in the header). Type your messages and press Enter. Type `q` to quit back to the menu.

Each session gets a unique ID (e.g. `bob_a3f9c1d2`) that ties together the save file and the long-term memory store. Two separate users starting a new chat with the same persona receive different IDs and fully isolated memories.

Debug lines are printed during the conversation to show the emotion pipeline:

```
[Debug] User Emotion: joy, Intensity: 0.81
[Debug] Bot Emotion: neutral
[LTM] Retrieved 3 memories.
Bob: Oh, pasta again? Sure, whatever.
```

### 2. Choose Personality

Lists all available personalities and lets you switch the active one before starting a chat. The change takes effect on the next *Start New Chat*.

### 3. Create Personality

Interactive wizard to define a new character:

- **Name** — the bot's display name.
- **Background** — a free-text description of the character's story, preferences, and quirks. The richer this is, the more consistent the character's behaviour.
- **OCEAN traits** — enter `high` or `low` for each of the five dimensions:
  - Openness
  - Conscientiousness
  - Extraversion
  - Agreeableness
  - Neuroticism
- **Initial comfortability** — whether the bot starts comfortable with the user (`y`) or not (`n`).

The profile is saved to `personalities.json`.

### 4. Delete Personality

Lists personalities and lets you remove one. The built-in `Bob` profile cannot be deleted (it is the fallback default). If you delete the currently active personality, the bot reverts to Bob automatically.

### 5. Save Chat

Saves the current session's state to `chat_saves/<chat_id>.json`. The long-term memory (ChromaDB) is stored on disk continuously and does not need to be explicitly saved — only the lightweight metadata (turn counter, working memory, reflection text, comfortability) is written here.

Re-saving an ongoing session updates only its own file. Different sessions are never overwritten.

### 6. Load Chat

Lists all saves in `chat_saves/`, sorted by most-recent turn first:

```
--- Saved Chats ---
  1. [Bob]   turn 24  –  id: bob_a3f9c1d2
  2. [Alice]  turn 9  –  id: alice_3c71e804
```

Selecting a save restores the full state and reopens the matching ChromaDB store so long-term memories from the previous session are immediately available.

### 7. Delete Saved Chat

Same picker as Load. Selecting a save permanently deletes both the JSON file and the ChromaDB folder for that session.

---

## Personality system

Personalities are built from two components:

**Background** — a plain-English description injected directly into every prompt. Include the character's age, occupation, likes, dislikes, speech style, and any other traits you want to be consistently reflected.

**OCEAN traits** — each dimension is mapped to a detailed natural-language description that is also injected into the prompt:

| Trait | Low | High |
|---|---|---|
| Openness | Practical, conventional, avoids creative ideas | Imaginative, curious, appreciates art |
| Conscientiousness | Impulsive, messy, avoids difficult tasks | Organised, self-disciplined, goal-driven |
| Extraversion | Introverted, prefers solo activities | Sociable, assertive, energetic |
| Agreeableness | Selfish, distrustful, holds grudges | Empathetic, cooperative, forgiving |
| Neuroticism | Calm, emotionally stable, patient | Anxious, emotionally reactive, tense |

**Comfortability** is a boolean flag updated every `WM_BUFFER_SIZE` turns. It reflects whether the bot currently feels at ease with this specific user and influences both its emotional responses and the tone of its replies.

---

## Memory system

### Working memory

A rolling window of the last `WM_BUFFER_SIZE` turns (default: 3). Each entry stores the user text, user emotion and intensity, bot emotion, and bot reply. Passed verbatim into every prompt.

### Reflection

Every `WM_BUFFER_SIZE` turns, Qwen generates a short internal monologue (4–5 sentences) summarising the bot's feelings and insights about the ongoing conversation. This replaces the previous reflection and is persisted in the save file. It is also written into long-term memory as a `reflection`-type entry.

### Long-term memory

Each session has a dedicated ChromaDB persistent store under `./long_memory/<chat_id>/`. Two things are stored:

- **Turn memories** — a one-line summary of each turn, written immediately after the bot replies.
- **Reflection memories** — the full reflection text, written every `WM_BUFFER_SIZE` turns.

Before each response, the `LTM_K` most relevant memories are retrieved using a combined score:

```
combined = (1 - alpha) * semantic_score + alpha * recency_score
recency  = exp(-decay * (current_turn - memory_turn))
```

`LTM_RECENCY_ALPHA = 0.3` means relevance dominates but recent turns get a moderate boost. Increase it toward `1.0` to bias strongly toward recency; decrease it toward `0.0` for purely semantic retrieval.

---

## Troubleshooting

**`ollama: command not found`**
Ollama is not on your PATH. On Linux, add `/usr/local/bin` to your PATH or restart your shell after installation.

**`Connection refused` when calling Ollama**
The daemon is not running. Start it with `ollama serve` in a separate terminal.

**EmoBART download fails**
You need a Hugging Face account for gated models. `lzw1008/Emobart-large` is public, so a plain `huggingface-cli login` is not required — check your internet connection and try again.

**CUDA out of memory**
Reduce the model size or set `device_map='cpu'` in `emotion_recognizer_bart.py`. Alternatively, run EmoBART on CPU and let Ollama handle GPU allocation for the chat model.

**Bot responses contain the raw prompt or reasoning tokens**
This happens when `think=False` is not supported by an older Ollama version. Update Ollama to the latest release: `ollama update`.

**`ValueError: Openness must be in ['High', 'Low']`**
A personality JSON file has a trait value other than `"high"` or `"low"` (case-insensitive). Edit `personalities.json` and correct the offending value.