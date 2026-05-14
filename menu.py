'''
Module that implements the following functionalities:
- Start a new chat
- Select a personality (from created ones)
- Create a new personality profile
- Delete a personality profile
- Save a chat
- Load a chat
- Delete a saved chat
'''
import json
import os
from bot import PEACh
from personality import PersonalityProfile, OCEANTraitDescription
from long_term_memory import LongTermMemory
from configs import CHAT_SAVES_DIR

current_bot_instance = None
current_personality = None
ocean_desc = OCEANTraitDescription()

def _save_path(chat_id: str) -> str:
    return os.path.join(CHAT_SAVES_DIR, f'{chat_id}.json')

def _list_saves() -> list[dict]:
    '''Return all parsed save dicts found in CHAT_SAVES_DIR, sorted by turn descending.'''
    if not os.path.exists(CHAT_SAVES_DIR):
        return []
    saves = []
    for fname in sorted(os.listdir(CHAT_SAVES_DIR)):
        if fname.endswith('.json'):
            fpath = os.path.join(CHAT_SAVES_DIR, fname)
            try:
                with open(fpath, 'r') as f:
                    saves.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                pass
    saves.sort(key=lambda s: s.get('turn', 0), reverse=True)
    return saves
 
 
def _print_saves(saves: list[dict]) -> None:
    for i, s in enumerate(saves, 1):
        print(f'  {i}. [{s.get('personality_name', '?')}]  '
              f'turn {s.get('turn', '?')}  –  id: {s.get('chat_id', '?')}')


def load_personalities_from_file():
    '''
        Loads personalities from a JSON file, or creates a default if missing.
    '''
    if os.path.exists('personalities.json'):
        with open('personalities.json', 'r') as f:
            return json.load(f)
    return {
        'Bob': {
            'name': 'Bob',
            'background': '15 years old highschool student. Likes pasta with tomato sauce. Dislikes dogs.',
            'openness': 'low', 'conscientiousness': 'high', 'extraversion': 'low', 
            'agreeableness': 'low', 'neuroticism': 'high', 'comfortability': True
        }
    }

def start_new_chat(save_state=None):
    '''
        Start a new chat with the selected personality, or the default one
    '''
    global current_bot_instance, current_personality
    print('\nLoading PEACh...')
    
    current_bot_instance = PEACh(personality=current_personality, save_state=save_state)
    current_bot_instance.chat()

def save_chat():
    '''
        Saves the chat:
        - Turn number
        - 3 recent turns
        - Thoughts on the user
        - Personality used
        - Comfortability of the bot
        - Chat id for ltm
    '''
    global current_bot_instance
    if not current_bot_instance:
        print('\nNo active chat to save! Start a chat first.')
        return
    
    os.makedirs(CHAT_SAVES_DIR, exist_ok=True)
    
    chat_data = {
        'turn': current_bot_instance.turn,
        'recent_turns': current_bot_instance.recent_turns,
        'thoughts_on_user': current_bot_instance.thoughts_on_user,
        'personality_name': current_bot_instance.personality.name,
        'comfortability': current_bot_instance.personality.comfortability,
        'chat_id': current_bot_instance.chat_id
    }
    
    path = _save_path(current_bot_instance.chat_id)
    with open(path, 'w') as f:
        json.dump(chat_data, f, indent=4)
    print(f'\nChat saved successfully to {path}')

def load_chat():
    '''
        Loads a saved chat:
        - Turn number
        - 3 recent turns
        - Thoughts on the user
        - Personality used
        - Comfortability of the bot
        - Chat id for ltm
    '''
    saves = _list_saves()
    if not saves:
        print('\nNo saved chats.')
        return

    print('\n--- Saved Chats ---')
    _print_saves(saves)

    choice = input('\nSelect a save number to load (or \'c\' to cancel): ')
    if not (choice.isdigit() and 1 <= int(choice) <= len(saves)):
        print('\nCancelled.')
        return
 
    global current_personality
    chat_data = saves[int(choice) - 1]
    profiles = load_personalities_from_file()
    p_name = chat_data.get('personality_name')

    
    if p_name in profiles:
        p_data = profiles[p_name]
        current_personality = PersonalityProfile(
            p_data['name'], p_data['background'], p_data['openness'], 
            p_data['conscientiousness'], p_data['extraversion'], 
            p_data['agreeableness'], p_data['neuroticism'], p_data.get('comfortability', True), 
            ocean_desc
        )
        print(f'\nLoaded saved chat with {p_name}.')
    else:
        print(f'\nWarning: Saved personality \'{p_name}\' not found. Using default.')
        current_personality = None

    start_new_chat(save_state=chat_data)

def delete_chat():
    '''
        Deletes a saved chat. Also ltm.
    '''
    global current_bot_instance
    saves = _list_saves()
    if not saves:
        print('\nNo saved chats found.')
        return
 
    print('\n--- Saved Chats ---')
    _print_saves(saves)
 
    choice = input('\nSelect a save number to delete (or \'c\' to cancel): ')
    if not (choice.isdigit() and 1 <= int(choice) <= len(saves)):
        print('\nCancelled.')
        return
 
    chat_data = saves[int(choice) - 1]
    chat_id = chat_data.get('chat_id', '')
 
    confirm = input(f'\nDelete save \'{chat_id}\'? This cannot be undone. (y/n): ').lower()
    if confirm != 'y':
        print('\nDeletion cancelled.')
        return
 
    if chat_id:
        success = LongTermMemory.delete_store(chat_id)
    
    if success:
        path = _save_path(chat_id)
        if os.path.exists(path):
            os.remove(path)
    
        print(f'\nSaved chat {chat_id} deleted.')


def choose_personality():
    '''
        Choose a personality from the saved ones.
    '''
    global current_personality
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
            ocean_desc
        )
        print(f'\nSuccessfully switched personality to {selected}!')
    else:
        print('\nSelection cancelled.')

def create_personality():
    '''
        Create a new personality form scratch and save it.
        - Name
        - Background
        - OCEAN
        - Comfortability at beginning
    '''
    print('\n--- Create New Personality ---')
    name = input('Name: ')
    background = input('Background/Description: ')
    
    print('\nFor the following OCEAN traits, enter \'high\' or \'low\'.')
    o = input('Openness: ').lower()
    c = input('Conscientiousness: ').lower()
    e = input('Extraversion: ').lower()
    a = input('Agreeableness: ').lower()
    n = input('Neuroticism: ').lower()

    traits = [o, c, e, a, n]
    if any(t not in ['high', 'low'] for t in traits):
        print('\nError: All traits must be exactly \'high\' or \'low\'. Creation failed.')
        return

    comf_input = input('Starts comfortable with user? (y/n): ').lower()
    if comf_input == 'y':
        comfortability = True
    elif comf_input == 'n':
        comfortability = False
    else:
        print('\nError: Comfort at beginning must be \'y\' or \'n\'. Creation failed.')
        return
    
    profiles = load_personalities_from_file()
    profiles[name] = {
        'name': name, 'background': background,
        'openness': o, 'conscientiousness': c, 
        'extraversion': e, 'agreeableness': a, 'neuroticism': n,
        'comfortability': comfortability
    }
    
    with open('personalities.json', 'w') as f:
        json.dump(profiles, f, indent=4)
        
    print(f'\nPersonality \'{name}\' created and saved successfully!')

def delete_personality():
    '''
        Deletes a saved personality.
    '''
    global current_personality, current_bot_instance
    
    profiles = load_personalities_from_file()
    names = list(profiles.keys())
    
    print('\n--- Delete Personality ---')
    for i, name in enumerate(names, 1):
        print(f'{i}. {name}')
        
    choice = input('\nSelect a personality number to delete (or \'c\' to cancel): ')
    if choice.isdigit() and 1 <= int(choice) <= len(names):
        selected = names[int(choice) - 1]
        
        if selected == 'Bob':
            print('\nError: You cannot delete the default \'Bob\' personality.')
            return
            
        confirm = input(f'\nAre you sure you want to delete \'{selected}\'? (y/n): ').lower()
        if confirm == 'y':
            del profiles[selected]
            
            with open('personalities.json', 'w') as f:
                json.dump(profiles, f, indent=4)
                
            print(f'\nPersonality \'{selected}\' deleted successfully!')
            
            # If the user deleted the currently active bot, reset back to Bob
            if current_personality and current_personality.name == selected:
                print('Active personality was deleted. Reverting to default (Bob).')
                p_data = profiles['Bob']
                current_personality = PersonalityProfile(
                    p_data['name'], p_data['background'], p_data['openness'], 
                    p_data['conscientiousness'], p_data['extraversion'], 
                    p_data['agreeableness'], p_data['neuroticism'], 
                    p_data.get('comfortability', True), ocean_desc
                )
                current_bot_instance = None # Clear the active bot instance
        else:
            print('\nDeletion cancelled.')
    else:
        print('\nSelection cancelled.')

def main_menu():
    '''
        Calls the functions and gives a minimal UI.
    '''
    global current_personality
    
    # Ensure default personality is loaded on startup if none is selected
    if not current_personality:
        profiles = load_personalities_from_file()
        p_data = profiles.get('Bob')
        current_personality = PersonalityProfile(
            p_data['name'], p_data['background'], p_data['openness'], 
            p_data['conscientiousness'], p_data['extraversion'], 
            p_data['agreeableness'], p_data['neuroticism'], 
            p_data.get('comfortability', True), ocean_desc
        )

    while True:
        print('\n' + '='*30)
        print(f'       PEACh Main Menu')
        print(f'       Active Bot: {current_personality.name}')
        print('='*30)
        print('1. Start New Chat')
        print('2. Choose Personality')
        print('3. Create Personality')
        print('4. Delete Personality')
        print('5. Save Chat')
        print('6. Load Chat')
        print('7. Delete Saved Chat')
        print('8. Exit')
        
        choice = input('\nSelect an option (1-8): ')
        
        if choice == '1':
            start_new_chat()
        elif choice == '2':
            choose_personality()
        elif choice == '3':
            create_personality()
        elif choice == '4':
            delete_personality()
        elif choice == '5':
            save_chat()
        elif choice == '6':
            load_chat()
        elif choice == '7':
            delete_chat()
        elif choice == '8':
            print('Exiting PEACh. Goodbye!')
            break
        else:
            print('Invalid choice. Please try again.')