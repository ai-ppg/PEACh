'''
Long-term memory backed by ChromaDB with Ollama embeddings.

Each chat session gets its own isolated persistent store in ./long_memory/<chat_id>/

A chat_id is generated as  <persona_name>_<8-hex-uuid> so two users
chatting with the same persona get completely separate stores.

Retrieval combines semantic similarity (cosine) with an exponential
recency decay so that older memories are progressively down-weighted:

    combined = (1 - alpha) * semantic_score + alpha * recency_score
    recency_score = exp(-decay * (current_turn - memory_turn))

alpha and decay are tunable in configs.py (LTM_RECENCY_ALPHA, LTM_RECENCY_DECAY).
'''

import os
import uuid
import math
import shutil
from datetime import datetime
import chromadb
from ollama import embeddings
from configs import CHROMADB_COLLECTION, EMBEDDING_MODEL_NAME, LTM_K, LTM_RECENCY_ALPHA, LTM_RECENCY_DECAY


class LongTermMemory:
    def __init__(self, chat_id: str):
        '''
            Initialise (or reopen) the persistent ChromaDB store for this chat.

            Args:
                chat_id: Unique identifier for the chat session, e.g. "bob_a3f9c1d2".
                        Use LongTermMemory.generate_chat_id() to create one.
        '''
        self.chat_id = chat_id
        self.store_path = os.path.join(CHROMADB_COLLECTION, chat_id)
        os.makedirs(self.store_path, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.store_path)
        self.collection = self.client.get_or_create_collection(
            name='memories',
            metadata={'hnsw:space': 'cosine'}
        )

    def _embed(self, text: str) -> list[float]:
        response = embeddings(model=EMBEDDING_MODEL_NAME, prompt=text)
        return response['embedding']

    def add_memory(self, text: str, turn: int) -> None:
        '''
            Embed and store a memory.

            Args:
                text:        The text to remember (turn summary or reflection).
                turn:        The conversation turn number (used for recency scoring).
                memory_type: "turn" for per-turn facts, "reflection" for periodic reflections.
        '''
        self.collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[self._embed(text)],
            documents=[text],
            metadatas=[{
                'turn':      turn,
                'timestamp': datetime.now().isoformat()
            }]
        )

    def retrieve_memories(
        self,
        query: str,
        current_turn: int,
        k: int       = LTM_K,
        alpha: float = LTM_RECENCY_ALPHA,
        decay: float = LTM_RECENCY_DECAY,
    ) -> list[str]:
        '''
        Return the top-k memories ranked by a combined semantic + recency score.

        Strategy:
            1. Fetch up to k*3 candidates via semantic (cosine) search.
            2. Re-rank each candidate with:
                   combined = (1-alpha)*semantic + alpha*recency
               where recency = exp(-decay * (current_turn - memory_turn)).
            3. Return the top-k texts.

        Args:
            query:        The current user utterance used as the search query.
            current_turn: Current turn number for recency calculation.
            k:            Number of memories to return.
            alpha:        Blend weight - 0 = pure semantic, 1 = pure recency.
            decay:        Exponential decay rate per turn.

        Returns:
            List of at most k memory strings, best-first.
        '''
        total = self.collection.count()
        if total == 0:
            return []

        fetch_n = min(total, k * 3)

        results = self.collection.query(
            query_embeddings=[self._embed(query)],
            n_results=fetch_n,
            include=['documents', 'distances', 'metadatas']
        )

        scored: list[tuple[float, str]] = []
        for doc, dist, meta in zip(
            results['documents'][0],
            results['distances'][0],
            results['metadatas'][0],
        ):
            semantic_score = 1.0 - dist
            memory_turn    = meta.get('turn', 0)
            recency_score  = math.exp(-decay * max(0, current_turn - memory_turn))
            combined       = (1.0 - alpha) * semantic_score + alpha * recency_score
            scored.append((combined, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:k]]

    def count(self) -> int:
        '''Return the number of memories currently stored.'''
        return self.collection.count()

    @staticmethod
    def generate_chat_id(persona_name: str) -> str:
        '''
        Create a unique, filesystem-safe chat identifier.

        Format: <persona_name_lower>_<8-char hex uuid>
        Example: "bob_a3f9c1d2"

        Two users chatting with the same persona will each call this function
        independently and receive different IDs, keeping their memories isolated.
        '''
        safe_name  = persona_name.replace(' ', '_').lower()
        short_uuid = uuid.uuid4().hex[:8]
        return f'{safe_name}_{short_uuid}'

    @staticmethod
    def delete_store(chat_id: str) -> None:
        '''
        Permanently remove the ChromaDB store for a given chat_id.
        Called by the menu when the user deletes a saved chat.
        '''
        path = os.path.join(CHROMADB_COLLECTION, chat_id)
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                print(f'[LTM] Deleted memory store: {chat_id}')
                return True
            except:
                print('\nThe chat you are trying to delete is still in memory. \nRe-run the script and then delete.')
                return False
        else:
            print(f'[LTM] No memory store found for: {chat_id}')
            return False