'''

'''
from emotion_recognizer_bart import get_emobart_tokenizer_and_model, get_user_emotional_status
from emotion_generator import generate_bot_emotion, generate_bot_comfortability
from personality import PersonalityProfile, OCEANTraitDescription
from working_memory import recent_turns_update, generate_bot_reflection
from long_term_memory import LongTermMemory
from response_generator import generate_bot_response
from configs import WM_BUFFER_SIZE

class PEACh:
    def __init__(self, personality: PersonalityProfile = None, save_state: dict = None):
        self.tokenizer, self.model = get_emobart_tokenizer_and_model()
        ocean_desc = OCEANTraitDescription()

        if personality:
            self.personality = personality
        else:
            self.personality = PersonalityProfile(
                'Bob', 
                '15 years old highschool student. Likes pasta with tomato sauce. Dislikes dogs.', 
                'low', 'high', 'low', 'low', 'high', True, ocean_desc
            )
        
        if save_state:
            self.thoughts_on_user = save_state.get('thoughts_on_user', '')
            self.recent_turns = save_state.get('recent_turns', [])
            self.turn = save_state.get('turn', 1)
            self.chat_id = save_state.get('chat_id')
            if 'comfortability' in save_state:
                self.personality.comfortability = save_state['comfortability']
            print(f"\n[Debug] Resuming chat n {self.chat_id} at turn {self.turn}")
        else:
            self.thoughts_on_user = str()
            self.recent_turns = list()
            self.turn = 1
            self.chat_id = LongTermMemory.generate_chat_id(self.personality.name)
            print(f'\n[Debug] New chat started (id: {self.chat_id})')
        
        self.ltm = LongTermMemory(self.chat_id)
        print(f'[LTM] Store loaded - {self.ltm.count()} memories.')


    def chat(self):
        print(f"\n--- Chat started with {self.personality.name} ---")
        print("(Type 'q' to quit and return to the main menu)\n")
        
        while True:
            print('\n' + '-'*20 + f' TURN {self.turn} ' + '-'*20 + '\n')
            
            utterance = input('You: ')
            if utterance.lower() == 'q':
                break
            user_emotion, user_emotion_intensity = get_user_emotional_status(self.tokenizer, self.model, utterance)
            print(f"[Debug] User Emotion: {user_emotion}, Intensity: {user_emotion_intensity:.2f}")
            bot_emotion = generate_bot_emotion(utterance, user_emotion, user_emotion_intensity, self.personality, self.thoughts_on_user, self.recent_turns)
            print(f"[Debug] Bot Emotion: {bot_emotion}")

            relevant_memories = self.ltm.retrieve_memories(utterance, self.turn)
            print(f'[LTM] Retrieved {len(relevant_memories)} memories.')


            bot_text = generate_bot_response(utterance, user_emotion, user_emotion_intensity, self.personality, self.thoughts_on_user, self.recent_turns, relevant_memories, bot_emotion)
            print(f'{self.personality.name}:', bot_text)

            self.recent_turns = recent_turns_update(utterance, user_emotion, user_emotion_intensity, bot_emotion, bot_text, self.recent_turns)
            turn_summary = (
                f'Turn {self.turn} - '
                f'User ({user_emotion}, {user_emotion_intensity:.2f}): {utterance} | '
                f'Bot ({bot_emotion}): {bot_text}'
            )
            self.ltm.add_memory(turn_summary, self.turn)


            if self.turn % WM_BUFFER_SIZE == 0:
                self.thoughts_on_user = generate_bot_reflection(self.personality, self.thoughts_on_user, self.recent_turns)
                self.personality.comfortability = generate_bot_comfortability(self.personality, self.thoughts_on_user, self.recent_turns)
                print('\n[Debug] Bot Reflection: ', self.thoughts_on_user)
                print('[Debug] Bot Comfort: ', self.personality.comfortability)
            
            self.turn += 1