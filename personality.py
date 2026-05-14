'''
Personality of the bot is given by the mix of name and background, with OCEAN traits and comfortability.
The personality class sotres those info.

The OCEANTraitsDescription is used only as an archive of OCEAN descriptions.
Those are used for emotional generation and reply generation.
'''
from dataclasses import dataclass

class OCEANTraitDescription:
    '''
        OCEAN personality trait description. They were obtained by NotebookLM analysis of the papers i gave it (all in the Papers folder).
    '''
    def __init__(self):

        self.ocean_descriptions = {
            'High': { 
                'Openness': 'Open-minded, imaginative, and sensitive, with a deep appreciation for artistic beauty, a tendency to actively seek knowledge across diverse fields, and a fascination with unconventional ideas.', 
                'Conscientiousness': 'Scrupulous, well-organized, and self-disciplined, maintaining an orderly environment, pursuing goals diligently, striving for excellence, and deliberating carefully before making decisions.', 
                'Extraversion': 'Friendly, assertive, energetic, and talkative, enjoying social interactions, feeling confident leading groups, and frequently experiencing positive emotions.', 
                'Agreeableness': 'Trusting, sympathetic, cooperative, and good-natured, showing a willingness to compromise, acting altruistically, and readily forgiving those who have wronged them.', 
                'Neuroticism': 'Prone to psychological distress, frequently feeling tense, anxious, angry, depressed, impulsive, and emotionally unstable.'
            }, 
            'Low': { 
                'Openness': 'Practical, realistic, and close-minded, often finding little enjoyment in art, experiencing minimal intellectual curiosity, and avoiding creative or unconventional activities.', 
                'Conscientiousness': 'Impulsive, unreliable, and messy, often avoiding challenging tasks, tolerating minor errors, and making decisions with little reflection.', 
                'Extraversion': 'Introverted, independent, and timid, often focusing more on solo activities, feeling uncomfortable as the center of attention, and being generally indifferent to social interactions.', 
                'Agreeableness': 'Selfish, distrustful, rude, and uncooperative, tending to hold grudges, criticize others, stubbornly defend personal views, and easily become angry.', 
                'Neuroticism': 'Emotionally stable, relaxed, calm, patient, easygoing, happy, and level-headed.'
            }
        }

@dataclass
class PersonalityProfile:
    '''
        Name and background of the Charactyer that the bot is going to interpret.

        Big Five personality vector. Each dimension: Low - High.
        O  Openness            Low = conventional                     High = creative/curious
        C  Conscientiousness   Low = spontaneous                      High = organised/disciplined
        E  Extraversion        Low = introverted                      High = extroverted/sociable
        A  Agreeableness       Low = unempathetic/competitive         High = empathetic/cooperative
        N  Neuroticism         Low = emotionally stable               High = anxious/reactive

        Comfortability: bool (True if comf, else False) -> if the bot feels comfortable or not in the interaction.
    '''
    name: str
    background: str
    openness: str
    conscientiousness: str
    extraversion: str
    agreeableness: str
    neuroticism: str
    comfortability: bool
    traits_description: dict

    def __post_init__(self):
        self.traits = {
            'Openness': self.openness.lower().capitalize(),
            'Conscientiousness': self.conscientiousness.lower().capitalize(),
            'Extraversion': self.extraversion.lower().capitalize(),
            'Agreeableness': self.agreeableness.lower().capitalize(),
            'Neuroticism': self.neuroticism.lower().capitalize()
        }
        
        for trait, value in self.traits.items():
            if not value in ['High', 'Low']:
                raise ValueError(f'{trait} must be in [\'High\', \'Low\']. Got {value}.')

        for trait, value in self.traits.items():
            trait = trait.lower()
            setter_name = f'_{trait}'
            setter = getattr(self, setter_name)
            self.trait = setter(value)

    def _openness(self, value: str):
        self.openness = self.traits_description.ocean_descriptions[value]['Openness']
    
    def _conscientiousness(self, value: str):
        self.conscientiousness = self.traits_description.ocean_descriptions[value]['Conscientiousness']
    
    def _extraversion(self, value: str):
        self.extraversion = self.traits_description.ocean_descriptions[value]['Extraversion']
    
    def _agreeableness(self, value: str):
        self.agreeableness = self.traits_description.ocean_descriptions[value]['Agreeableness']
    
    def _neuroticism(self, value: str):
        self.neuroticism = self.traits_description.ocean_descriptions[value]['Neuroticism']
    
    def set_comfortability(self, is_comfortable: bool):
        self.comfortability = is_comfortable
