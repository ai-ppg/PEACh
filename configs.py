'''
This bot has been tested with Qwen3.5-4B. The parameters are chosen according to the model specifications.
When changing the model, pleas change also the parameters, based on the task.
Parameter values taken from https://huggingface.co/Qwen/Qwen3.5-4B

LLM used is Qwen2.5-4B via Ollama.
@misc{qwen3.5,
    title  = {{Qwen3.5}: Towards Native Multimodal Agents},
    author = {{Qwen Team}},
    month  = {February},
    year   = {2026},
    url    = {https://qwen.ai/blog?id=qwen3.5}}
'''

MODEL_NAME = 'qwen3.5:4b'

RESPONSE_PARAMS = {       
                'temperature': 0.6,
                'top_p': 0.95,
                'top_k': 20,
                'presence_penalty': 1.5
            }
REFLECTION_PARAMS = {       
                'temperature': 1.0,
                'top_p': 0.95,
                'top_k': 20,
                'presence_penalty': 1.5
            }  
EMOTIONAL_STATE_PARAMS = {
                'temperature': 0.3,
                'top_p': 0.95,
                'top_k': 20,
                'presence_penalty': 1.5,
            }

EMBEDDING_MODEL_NAME = 'embeddinggemma'
LTM_K = 4
LTM_RECENCY_ALPHA = 0.3
LTM_RECENCY_DECAY = 0.05

EMOBART_MODEL_ID = "lzw1008/Emobart-large"
SAVE_EMOBART_DIR = "./local_emobart_model"
CHROMADB_COLLECTION = './long_memory'
CHAT_SAVES_DIR = './chat_saves'
TEST_FOLDER = './tests_results'

WM_BUFFER_SIZE = 3