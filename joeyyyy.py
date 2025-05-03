import speech_recognition as sr
import pyttsx3
import time
import random
import re
import os
import pygame
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator
from gtts import gTTS
from langdetect import detect

# === Joey's Brain (Intents) ===
intents = {
    "greet": [
        "hello", "hi", "hey", "good morning", "good evening", "what's up", "namaste", "namaste joey", "kem cho", "hola", "konnichiwa", "guten tag", "bonjour", "ciao", "say hello to",
        # More English greetings
        "greetings", "hiya", "howdy", "good afternoon",
        # More Spanish greetings
        "buenos dias", "buenas tardes", "buenas noches", "¿qué tal?", "¿cómo estás?",
        # More Hindi greetings
        "नमस्ते", "सुप्रभात", "শুভ সন্ধ্যা", "क्या हाल है?",
        # More Urdu greetings
        "اسلام علیکم", "آداب", "صبح بخیر", "شام بخیر",
        # More Bangla greetings
        "নমস্কার", "আসসালামু আলাইকুম", "শুভ সকাল", "শুভ সন্ধ্যা",
        # Japanese greetings
        "こんにちは", "もしもし", "おはようございます", "こんばんは", "やあ",
        # Korean greetings
        "안녕하세요", "여보세요", "좋은 아침", "좋은 저녁", "잘 지내세요?"
    ],
    "ask for help": [
        "what can you do", "help me", "how can you assist", "what else can you do", "what are your features",
        # More English help requests
        "can you help?", "assist me please", "what services do you offer?",
        # More Spanish help requests
        "¿en qué puedes ayudarme?", "ayúdame por favor", "¿qué servicios ofreces?",
        # More Hindi help requests
        "आप क्या कर सकते हैं?", "मेरी मदद करो", "आप कैसे सहायता कर सकते हैं?",
        # More Urdu help requests
        "آپ کیا کر سکتے ہیں؟", "میری مدد کریں", "آپ کیسے مدد کر سکتے ہیں؟",
        # More Bangla help requests
        "আপনি কি করতে পারেন?", "আমাকে সাহায্য করুন", "আপনি কিভাবে সাহায্য করতে পারেন?",
        # Japanese help requests
        "何ができますか？", "助けてください", "どうすればお手伝いできますか？",
        # Korean help requests
        "무엇을 할 수 있나요?", "도와주세요", "어떻게 도와드릴까요?"
    ],
    "emergency call": [
        "call someone", "call emergency", "help me", "emergency", "distress", "urgent", "call police",
        # More English emergency
        "I need help", "emergency assistance", "call for help", "urgent help", "I'm in danger",
        # Spanish emergency
        "necesito ayuda", "emergencia", "ayuda urgente", "llama a la policía", "estoy en peligro",
        # Hindi emergency
        "मुझे मदद चाहिए", "आपातकालीन", "तुरंत मदद", "पुलिस को बुलाओ", "मैं खतरे में हूँ",
        # Urdu emergency
        "مجھے مدد کی ضرورت ہے", "ایمرجنسی", "فوری مدد", "پولیس کو بلاؤ", "میں خطرے میں ہوں",
        # Bangla emergency
        "আমার সাহায্য দরকার", "জরুরী", "তাৎক্ষণিক সাহায্য", "পুলিশ ডাকো", "আমি বিপদে আছি",
        # Japanese emergency
        "助けて", "緊急", "助けが必要です", "警察を呼んで",
        # Korean emergency
        "도와줘", "긴급", "긴급 상황", "경찰 불러"
    ],
    "tell a joke": ["tell me a joke", "make me laugh", "say something funny", "give me a joke", "got any jokes?", "tell me something funny", "tell a joke in hindi", "tell a joke in spanish", "tell a joke in urdu", "tell a joke in bangla", "tell a joke in japanese", "tell a joke in korean"], # Added language variations for intent matching
    "joke feedback": ["not funny", "bad joke", "you are not funny", "that was terrible", "your jokes are bad", "try a different joke"],
    "thank you": [
        "thank you", "thanks", "appreciate it", "grateful", "dhanyawad", "arigato", "gracias", "merci", "grazie",
        # More English thank you
        "thanks a lot", "thank you very much", "cheers",
        # More Spanish thank you
        "muchas gracias",
        # More Hindi thank you
        "आपका धन्यवाद", "बहुत बहुत धन्यवाद",
        # More Urdu thank you
        "شکریہ", "بہت بہت شکریہ",
        # More Bangla thank you
        "ধন্যবাদ", "অনেক ধন্যবাদ",
        # Japanese thank you
        "ありがとう", "ありがとうございます",
        # Korean thank you
        "감사합니다", "고마워요"
    ],
    # Removed "hindi off" and similar from stop or exit
    "stop or exit": ["stop", "exit", "quit", "turn off", "bye", "shut down", "goodbye", "tata", "alvida"],
    "introduce myself": ["my name is", "i am", "i'm", "you can call me", "my name is called"],
    "ask location": [
        "where am i", "what city am i in", "my location", "where am i located",
        # More English location
        "current location", "tell me my location",
        # More Spanish location
        "¿dónde estoy?", "¿cuál es mi ubicación?",
        # More Hindi location
        "मैं कहाँ हूँ?", "मेरी लोकेशन क्या है?",
        # More Urdu location
        "میں کہاں ہوں؟", "میری لوکیشن کیا ہے؟",
        # More Bangla location
        "আমি কোথায়?", "আমার অবস্থান কি?",
        # Japanese location
        "私はどこですか？", "私の場所は？",
        # Korean location
        "저는 어디에 있나요?", "제 위치는요?"
    ],
    "tell time": [
        "what time is it", "tell me the time", "what's the time", "current time",
        # More English time
        "time now", "can you tell me the time?",
        # More Spanish time
        "¿qué hora es?", "dime la hora",
        # More Hindi time
        "समय क्या हुआ है?", "कितना बजा है?",
        # More Urdu time
        "वक्त کیا ہوا ہے؟", "کتنا بجا ہے؟",
        # More Bangla time
        "সময় কত?", "কটা বাজে?",
        # Japanese time
        "何時ですか？", "時間を教えて",
        # Korean time
        "몇 시예요?", "시간 알려줘"
    ],
    "ask weather": [
        "what's the weather", "tell me the weather", "how is the weather", "weather now",
        # More English weather
        "current weather", "weather forecast", "is it raining?", "is it sunny?",
        # More Spanish weather
        "¿cómo está el clima?", "¿cuál es el pronóstico del tiempo?", "¿está lloviendo?", "¿hace sol?",
        # More Hindi weather
        "मौसम कैसा है?", "আজ কা মৌসুম", "क्या बारिश हो रही है?", "क्या धूप है?", # Corrected Hindi transliteration
        # More Urdu weather
        "موسم کیسا ہے؟", "آج کا موسم", "کیا بارش ہو رہی ہے؟", "کیا دھوپ ہے؟",
        # More Bangla weather
        "আবহাওয়া কেমন?", "আজকের আবহাওয়া", "বৃষ্টি হচ্ছে?", "রোদ আছে?",
        # Japanese weather
        "天気はどうですか？", "今日の天気", "雨が降っていますか？", "晴れていますか？",
        # Korean weather
        "날씨 어때요?", "오늘 날씨", "비가 오고 있나요?", "맑은가요?"
    ],
    "translate": ["say this in hindi", "translate this to hindi", "convert to hindi",
                  "say this in spanish", "translate this to spanish",
                  "say this in urdu", "translate this to urdu",
                  "say this in bangla", "translate this to bangla",
                  "translate to hindi", "translate to spanish", "translate to urdu", "translate to bangla",
                  # Japanese translate
                  "say this in japanese", "translate this to japanese", "translate to japanese",
                  # Korean translate
                  "say this in korean", "translate this to korean", "translate to korean"],
    "ask name": [
        "what's my name", "who am i", "my name", # Keep these for asking user's name
        # Add phrases for asking Joey's name
        "what is your name", "who are you", "your name", # Added comma here
        # Japanese ask name
        "あなたの名前は何ですか？", "あなたは誰ですか？",
        # Korean ask name
        "이름이 뭐예요?", "당신은 누구세요?"
    ],
    # Added language mode commands to intents for better matching
    "language_mode": ["hindi on", "hindi mode", "spanish on", "spanish mode", "urdu on", "urdu mode", "bangla on", "bangla mode", "english on", "english mode", "japanese on", "japanese mode", "korean on", "korean mode"], # Added korean
    # New intents for self-referential questions
    "ask origin": ["who made you", "who created you", "who is your creator",
                   # Japanese ask origin
                   "誰があなたを作りましたか？", "あなたの生みの親は誰ですか？",
                   # Korean ask origin
                   "누가 당신을 만들었나요?", "당신의 창조자는 누구인가요?"],
    "ask appearance": ["what color is your hair", "do you have hair", "what do you look like", "are you a robot", "are you human",
                      # Japanese ask appearance
                      "髪の色は何ですか？", "髪はありますか？", "どんな見た目ですか？",
                      # Korean ask appearance
                      "머리색이 뭐예요?", "머리카락이 있나요?", "어떻게 생겼어요?"],
    # New intent for asking age
    "ask age": ["how old are you", "what is your age", "your age",
                # Japanese ask age
                "何歳ですか？", "あなたの年齢は？",
                # Korean ask age
                "몇 살이에요?", "나이가 어떻게 돼요?"],
    # New intent for asking vehicle info
    "ask vehicle info": ["tell me about my car", "car status", "vehicle status", "my car info", "how is my car", # English phrases
                         # Add language variations for intent matching
                         "tell me about my car in hindi", "car status in spanish", "vehicle status in urdu", "my car info in bangla", "how is my car in japanese", "tell me about my car in korean"],
    # New intents for riddles and facts
    "tell riddle": ["tell me a riddle", "ask me a riddle", "give me a riddle", "riddle time",
                    # Add language variations for intent matching
                    "tell me a riddle in hindi", "ask me a riddle in spanish", "give me a riddle in urdu", "riddle time in bangla", "tell me a riddle in japanese", "ask me a riddle in korean"],
    "tell fact": ["tell me a fact", "give me a fact", "random fact", "tell me something interesting", "fact time",
                  # Add language variations for intent matching
                  "tell me a fact in hindi", "give me a fact in spanish", "random fact in urdu", "tell me something interesting in bangla", "fact time in japanese", "tell me a fact in korean"]
}

jokes = [
    "Parallel lines have so much in common… it’s a shame they’ll never meet.",
    "I told my computer I needed a break, and now it won’t stop sending me beach pictures.",
    "Why don’t scientists trust atoms? Because they make up everything.",
    # More jokes
    "Did you hear about the mathematician who’s afraid of negative numbers? He’ll stop at nothing to avoid them.",
    "Why did the scarecrow win an award? Because he was outstanding in his field!",
    "I'm reading a book about anti-gravity. It's impossible to put down!",
    "What do you call a fish with no eyes? Fsh!",
    "Why did the bicycle fall over? Because it was two tired!"
]

riddles = [
    {"riddle": "I speak without a mouth and hear without ears. I have no body, but I come alive with wind. What am I?", "answer": "An echo"},
    {"riddle": "What has an eye, but cannot see?", "answer": "A needle"},
    {"riddle": "What is full of holes but still holds water?", "answer": "A sponge"},
    {"riddle": "What is always in front of you but can’t be seen?", "answer": "The future"},
    {"riddle": "What has a heart that doesn’t beat?", "answer": "An artichoke"}
]

facts = [
    "A single cloud can weigh more than 1 million pounds.",
    "The shortest war in history was between Britain and Zanzibar on August 27, 1896. Zanzibar surrendered after just 38 minutes.",
    "There are more stars in the universe than grains of sand on all the Earth's beaches.",
    "A group of owls is called a parliament.",
    "Honey never spoils. Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible."
]


# === Speech Engine Setup ===
engine = pyttsx3.init()
voices = engine.getProperty('voices')

# Set default voice to Hindi male or fallback to English
# Note: Voice availability can vary by system. This attempts to find a Hindi male voice.
hindi_male_found = False
for voice in voices:
    # Check for Hindi and male in the voice name (case-insensitive)
    if "hindi" in voice.name.lower() and "male" in voice.name.lower():
        engine.setProperty('voice', voice.id)
        hindi_male_found = True
        break

# If no Hindi male voice is found, try to find a common English male voice like David
if not hindi_male_found:
    for voice in voices:
        if "david" in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break
    # If still no voice is set, pyttsx3 will likely use a default system voice.

# Set the default speech rate for general conversation
default_rate = 180
engine.setProperty('rate', default_rate)

recognizer = sr.Recognizer()
user_name = None # Global variable to store the user's name
active_language = 'en' # Variable to store the current active language mode (default is English)

# Variables for simulated driving checks
last_speeding_warning_time = 0
last_red_light_warning_time = 0
speed_check_interval = 120 # seconds (2 minutes)
traffic_check_interval = 120 # seconds (2 minutes)
last_speed_check = time.time()
last_traffic_check = time.time()

# Setup for intent matching using TF-IDF and Cosine Similarity
vectorizer = TfidfVectorizer()
intent_phrases = []
intent_tags = []

for tag, phrases in intents.items():
    intent_phrases.extend(phrases)
    intent_tags.extend([tag] * len(phrases))

# Fit the vectorizer to the intent phrases
X = vectorizer.fit_transform(intent_phrases)

def speak(text, lang='en'):
    """
    Speaks the given text using either pyttsx3 (for English) or gTTS (for other languages).
    Handles basic error printing.
    """
    print(f"Joey ({lang}): {text}")
    if lang == 'en':
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[English Voice Error] {e}")
    else:
        play_audio(text, lang)

def play_audio(text, lang):
    """
    Uses gTTS to generate speech audio for non-English languages and plays it using pygame.
    Cleans up the temporary audio file.
    """
    try:
        tts = gTTS(text=text, lang=lang)
        file_path = f"response_{lang}.mp3"
        tts.save(file_path)
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            time.sleep(0.1) # Small sleep to prevent high CPU usage
        pygame.mixer.quit()
        os.remove(file_path) # Clean up the temporary file
    except Exception as e:
        print(f"[{lang.upper()} Voice Error] {e}")
        # Fallback to English speech if audio playback fails
        speak("Sorry, I couldn't play the audio in that language.")

def translate_and_speak(text, target_lang="hi"):
    """
    Translates text to the target language using GoogleTranslator and then speaks it.
    Includes a print statement for debugging.
    """
    try:
        print(f"[Attempting translation to {target_lang}: {text}]") # Debug print
        translated_text = GoogleTranslator(source='auto', target=target_lang).translate(text)
        print(f"Joey (Translated to {target_lang}): {translated_text}")
        play_audio(translated_text, target_lang)
    except Exception as e:
        speak("Sorry, I couldn't translate right now.")
        print(f"Translation error: {e}")

def detect_language(text):
    """
    Attempts to detect the language of the input text. Includes specific checks for
    known phrases and keywords before using the langdetect library.
    """
    try:
        print(f"[Attempting language detection for: {text}]")
        # Specific case handling for common foreign phrases
        if text.lower() == "como esta":
            print("[Specific case: 'como esta' detected as Spanish]")
            return "es"
        spanish_keywords = ["hola", "buenos dias", "buenas tardes", "buenas noches", "que tal", "gracias", "de nada"]
        if any(keyword in text.lower() for keyword in spanish_keywords):
            print("[Detected Spanish keyword, setting language to Spanish]")
            return "es"
        hindi_keywords = ["नमस्ते", "मुझे", "مدد", "چاہیے", "क्या हाल है", "आप कैसे हैं", "यह क्या है", "कैसा", "मौसम", "आज", "तापमान", "समय", "کहाँ", "मेरा नाम"]
        if any(keyword in text for keyword in hindi_keywords): # Check for Hindi script directly
             print("[Overriding language to Hindi due to keyword]")
             return "hi"
        # Check for common foreign greetings that might confuse langdetect
        common_foreign_greetings = ["konnichiwa", "guten tag", "bonjour", "ciao", "kem cho", "guten morgen"]
        if any(greeting in text.lower() for greeting in common_foreign_greetings):
             print("[Detected a foreign greeting, defaulting to English]")
             return "en" # Default to English for other greetings for now

        # Use langdetect as a general fallback
        lang = detect(text)
        print(f"[Detected User Language: {lang}]")
        return lang
    except:
        print("[Language Detection Failed] Defaulting to English.")
        return "en"

def match_intent(user_input):
    """
    Matches the user input to the closest intent using TF-IDF and Cosine Similarity.
    Returns the matched intent tag and the similarity score.
    """
    if not user_input: # Handle empty input
        return "none", 0.0
    user_vec = vectorizer.transform([user_input])
    scores = cosine_similarity(user_vec, X)
    best_match_idx = scores.argmax()
    best_score = scores[0, best_match_idx]
    return intent_tags[best_match_idx], best_score

def extract_name(user_input):
    """
    Extracts a potential name from the user input if they are introducing themselves.
    Updates the global user_name variable.
    """
    global user_name
    # Look for patterns like "my name is [name]", "i am [name]", "i'm [name]"
    match = re.search(r"(my name is|i am|i'm)\s+([a-zA-Z]+)", user_input.lower())
    if match:
        user_name = match.group(2).capitalize()
        return user_name
    # Also look for just a name after a potential greeting or question
    # This is a simpler check, might need refinement based on expected input after welcome
    name_only_match = re.match(r"^([a-zA-Z]+)$", user_input.lower().strip())
    if name_only_match:
         user_name = name_only_match.group(1).capitalize()
         return user_name

    return None

# === Simulated Functions (Replace with actual APIs if available) ===
def get_location():
    """Simulates getting the current location."""
    return "New Delhi, Delhi, India" # Hardcoded location

def tell_time():
    """Gets the current time."""
    now = datetime.now()
    return now.strftime("%I:%M %p") # Format: 11:01 PM

def get_weather():
    """Simulates getting the current weather with varied responses."""
    weather_conditions = [
        "The weather in New Delhi is currently pleasant with a temperature of 35°C and mostly sunny skies.",
        "It's a bit warm in New Delhi right now, around 38°C with clear skies.",
        "Currently, the weather in New Delhi is 32°C and partly cloudy.",
        "Expect temperatures around 36°C in New Delhi with a gentle breeze.",
        "The weather in New Delhi is sunny and hot, about 39°C."
        # Add more varied weather descriptions here if needed
    ]
    return random.choice(weather_conditions) # Return a random weather description


def get_current_speed():
    """Simulates getting the current vehicle speed."""
    return random.randint(30, 80) # Random speed between 30 and 80

def get_traffic_signal_status():
    """Simulates getting the current traffic light status."""
    # 80% chance of green, 20% chance of red
    return random.choices(['green', 'red'], weights=[4, 1])[0]

def simulate_heartbeat():
    """Simulates getting a user's heartbeat."""
    return random.randint(60, 100) # Random heartbeat between 60 and 100

# === Core Chatbot Logic ===
def listen():
    """
    Listens for user audio input using the microphone and converts it to text.
    Includes error handling for speech recognition issues.
    """
    with sr.Microphone() as source:
        print("Calibrating mic for ambient noise...")
        # Adjust for ambient noise for 2 seconds
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Listening...")
        # Listen for up to 25 seconds for a phrase (Increased listening time)
        audio = recognizer.listen(source, phrase_time_limit=25)
        print("Processing...")
        try:
            # Use Google Web Speech API for recognition
            text = recognizer.recognize_google(audio)
            print(f"You: {text}")
            return text.lower() # Return lowercased text
        except sr.UnknownValueError:
            print("Didn't catch that.")
            return "" # Return empty string if speech is unintelligible
        except sr.RequestError:
            print("Service unavailable.")
            return "" # Return empty string if the API is unreachable
        except Exception as e:
            print(f"Error in listen(): {e}")
            return "" # Return empty string for any other errors

def handle_emergency(lang):
    """Simulates handling an emergency situation."""
    # Translate the emergency message
    emergency_message = "Emergency situation detected. Calling emergency services now. Hold tight."
    if lang != 'en':
        translate_and_speak(emergency_message, lang)
    else:
        speak(emergency_message, lang)
    print(">>> Dialing 112...") # Placeholder for actual emergency call


def handle_distress_signal(user_input, score, user_lang):
    """
    Checks for distress signals in user input and initiates emergency handling
    if a high-confidence distress signal is detected.
    The interaction language is based on the active_language mode.
    """
    global active_language # Access the global active language mode

    distress_signals = ["help", "emergency", "urgent", "distress",
                        "ayuda", "emergencia", "urgente", # Spanish
                        "مदد", "آपातकालीन", "तुरंत", # Hindi
                        "مدد", "ایمرجنسی", "فوری", # Urdu
                        "সাহায্য", "জরুরী", "তাৎক্ষণিক", # Bangla
                        # Japanese distress signals
                        "助けて", "緊急", "助けが必要です",
                        # Korean distress signals
                        "도와줘", "긴급", "긴급 상황", "경찰 불러"
                       ]
    # Check if any distress signal is in the input and the intent score is high
    if any(signal in user_input for signal in distress_signals) and score > 0.7:
        # Reassuring message in the active language
        reassure_message = "I detect a distress signal. Please stay calm. I am initiating emergency procedures."
        if active_language != 'en':
             translate_and_speak(reassure_message, active_language)
        else:
             speak(reassure_message, active_language)

        handle_emergency(active_language) # Pass active_language to handle_emergency

        heartbeat = simulate_heartbeat()
        heartbeat_message = f"Your current heartbeat is {heartbeat} bpm."
        if active_language != 'en':
             translate_and_speak(heartbeat_message, active_language)
        else:
             speak(heartbeat_message, active_language)

        are_you_alright_message = "Are you alright?"
        if active_language != 'en':
             translate_and_speak(are_you_alright_message, active_language)
        else:
             speak(are_you_alright_message, active_language)

        user_response = listen() # Listen for user's response after emergency action
        if user_response:
            user_lang_response = detect_language(user_response) # Still detect language of response
            if "no" in user_response:
                take_breaths_message = "Take deep breaths."
                call_someone_message = "Would you like me to call someone?"
                # Speak follow-up in the active language
                if active_language != 'en':
                    translate_and_speak(take_breaths_message, active_language)
                    translate_and_speak(call_someone_message, active_language)
                else:
                    speak(take_breaths_message, active_language)
                    speak(call_someone_message, active_language)

                if "yes" in listen(): # Listen for confirmation to call someone
                    handle_emergency(active_language) # Pass active_language
            else:
                glad_okay_message = "Glad you're okay. I'm here if you need anything."
                # Speak follow-up in the active language
                if active_language != 'en':
                    translate_and_speak(glad_okay_message, active_language)
                else:
                    speak(glad_okay_message, active_language)
        else:
            no_response_message = "I didn't hear a response. Please let me know if you need help."
            # Speak follow-up in the active language
            if active_language != 'en':
                 translate_and_speak(no_response_message, active_language)
            else:
                 speak(no_response_message, active_language)

def check_and_warn_speeding(user_lang):
    """
    Periodically checks the simulated speed and warns the user if they are speeding.
    """
    global last_speed_check
    current_time = time.time()
    # Check if enough time has passed since the last speed check
    if current_time - last_speed_check >= speed_check_interval:
        current_speed = get_current_speed()
        if current_speed > 60:  # Assuming 60 is the speed limit
            # Warning is in English for now, as it's a critical driving alert
            speak("Warning! You are speeding. Please slow down!", lang='en')
        last_speed_check = current_time # Update the time of the last check

def check_and_warn_traffic_light(user_lang):
    """
    Periodically checks the simulated traffic light status and warns the user if it's red.
    """
    global last_traffic_check
    current_time = time.time()
    # Check if enough time has passed since the last traffic check
    if current_time - last_traffic_check >= traffic_check_interval:
        signal_status = get_traffic_signal_status()
        if signal_status == 'red':
            # Warning is in English for now, as it's a critical driving alert
            speak("Traffic light is red. Please stop!", lang='en')
        last_traffic_check = current_time # Update the time of the last check


def main():
    """
    The main loop for the chatbot. Listens for input, processes it, and responds.
    Includes checks for specific commands and general intents.
    """
    global active_language, last_red_light_warning_time, user_name, default_rate # Ensure necessary globals are accessible

    # --- Initial Interaction: Greeting and Name Capture ---
    # Temporarily set a slower rate for the initial greeting
    initial_rate = 160 # You can adjust this value
    engine.setProperty('rate', initial_rate)

    # Speak the initial greeting and ask for the user's name
    # Updated initial greeting text with commas
    speak("hello, i am joey, what is your name")

    # Restore the default speech rate after the initial greeting
    engine.setProperty('rate', default_rate)

    # Listen for the user's immediate response
    initial_response = listen()

    # Attempt to extract the name from the initial response
    captured_name = extract_name(initial_response)

    # If a name was captured, respond with a personalized welcome and offer assistance
    if captured_name:
        # Updated personalized welcome text with commas and assistance offer
        speak(f"hello {captured_name}, welcome to okay driver, how may i help you")
    # --- End of Initial Interaction ---


    last_red_light_warning_time = 0 # Initialize the traffic light warning time
    previous_user_lang = 'en' # Initialize with English

    while True:
        # Listen for user input (this is the main listening point after the initial interaction)
        user_input = listen()

        # If no input was detected, perform background checks and continue
        if not user_input:
            check_and_warn_speeding(previous_user_lang)
            check_and_warn_traffic_light(previous_user_lang)
            time.sleep(1) # Small delay to avoid busy-waiting
            continue # Skip to the next iteration of the loop

        # --- Specific Handling for Custom Boss Greetings ---
        # Check if the user input exactly matches one of the custom boss greeting phrases
        if user_input.lower() == "say hello to boss joey" or user_input.lower() == "say hello to our boss":
            speak("hello tushkit gupta") # Respond with the specific custom greeting
            continue # Skip the rest of the loop and listen for the next input
        # --- End of Custom Boss Greetings Handling ---

        # --- Language Mode Handling (On/Mode) ---
        # Check for language mode activation commands
        language_on_match = re.search(r"(hindi|spanish|urdu|bangla|bengali|english|japanese|korean)\s+(on|mode)", user_input.lower()) # Added korean
        if language_on_match:
            language_name = language_on_match.group(1).lower()
            # Map language names to language codes
            lang_code_map = {'hindi': 'hi', 'spanish': 'es', 'urdu': 'ur', 'bangla': 'bn', 'bengali': 'bn', 'english': 'en', 'japanese': 'ja', 'korean': 'ko'} # Added korean
            new_active_language = lang_code_map.get(language_name, 'en') # Default to English if unrecognized

            active_language = new_active_language # Set the new active language

            # Provide confirmation in the newly set language
            confirmation_messages = {
                'en': "Okay, English mode activated.",
                'hi': "ठीक है, हिंदी मोड चालू किया।",
                'es': "Okay, modo español activado.",
                'ur': "ٹھیک ہے، اردو موڈ فعال ہو گیا۔",
                'bn': "ঠিক আছে, বাংলা মোড চালু করা হয়েছে।",
                'ja': "はい、日本語モードが有効になりました。", # Japanese confirmation
                'ko': "네, 한국어 모드가 활성화되었습니다." # Korean confirmation
            }
            speak(confirmation_messages.get(active_language, confirmation_messages['en']), active_language)
            continue # Skip the rest of the loop and listen for the next input
        # --- End of Language Mode Handling (On/Mode) ---

        # --- Language Mode Handling (Off) ---
        # Check for language mode deactivation commands (switch back to English)
        language_off_match = re.search(r"(hindi|spanish|urdu|bangla|bengali|japanese|korean)\s+off", user_input.lower()) # Added korean
        if language_off_match:
             language_name_off = language_off_match.group(1).lower() # Get the language name mentioned
             active_language = 'en' # Always switch back to English

             # Provide confirmation in English
             speak(f"Okay, switching back to English mode.", 'en')
             continue # Skip the rest of the loop and listen for the next input
        # --- End of Language Mode Handling (Off) ---


        # --- General Intent Matching ---
        # This must happen AFTER checking for specific commands like language mode
        intent, score = match_intent(user_input)
        print(f"Matched Intent: {intent} (score: {round(score, 2)})")
        # --- End of General Intent Matching ---


        # Detect the language of the user's input (still useful for distress signals, etc.)
        user_lang = detect_language(user_input)
        print(f"[Using detected language: {user_lang}]")
        previous_user_lang = user_lang # Store the detected language for background checks

        # Determine the response language based on the active language mode
        response_lang = active_language


        response_text = "" # Initialize response text

        # Handle responses based on the matched intent
        if intent == "greet":
            # --- Handling for "say hello to [name] in [language]" ---
            # Use regex to find the pattern "say hello to" followed by a name, then "in" and a language
            say_hello_to_name_lang_match = re.search(r"say hello to\s+([a-zA-Z]+)\s+in\s+(hindi|spanish|urdu|bangla|bengali|japanese|korean)", user_input.lower()) # Added korean
            if say_hello_to_name_lang_match:
                name_to_greet = say_hello_to_name_lang_match.group(1).capitalize() # Extract and capitalize the name
                language = say_hello_to_name_lang_match.group(2).lower() # Extract the language
                # Map language names to language codes
                lang_code = {'hindi': 'hi', 'spanish': 'es', 'urdu': 'ur', 'bangla': 'bn', 'bengali': 'bn', 'japanese': 'ja', 'korean': 'ko'}.get(language, 'en') # Added korean

                if lang_code != 'en': # If a non-English language was requested
                     speak(f"Okay, let me say hello to {name_to_greet} in {language}.") # Acknowledge the translation request
                     if lang_code == 'es':
                          translated_greeting = f"¡Hola {name_to_greet}!"
                     elif lang_code == 'hi':
                          translated_greeting = f"नमस्ते {name_to_greet}!"
                     elif lang_code == 'ur':
                          translated_greeting = f"اسلام علیکم {name_to_greet}!" # Basic Urdu greeting
                     elif lang_code == 'bn':
                          translated_greeting = f"হ্যালো {name_to_greet}!" # Basic Bangla greeting
                     elif lang_code == 'ja':
                          translated_greeting = f"こんにちは {name_to_greet}さん！" # Basic Japanese greeting
                     elif lang_code == 'ko':
                          translated_greeting = f"안녕하세요 {name_to_greet}님!" # Basic Korean greeting
                     else:
                          translated_greeting = f"Hello {name_to_greet}!" # Fallback to English greeting for the greeting itself

                     speak(translated_greeting, lang_code)
                else: # If English was requested or language not recognized in the pattern
                     speak(f"Hello {name_to_greet}!") # Just say hello in English

                continue # Skip the rest of the greet handling

            # --- End of Handling for "say hello to [name] in [language]" ---

            # --- Handling for "say hello to [name]" ---
            say_hello_to_match = re.search(r"say hello to\s+([a-zA-Z]+)", user_input.lower())
            if say_hello_to_match:
                name_to_greet = say_hello_to_match.group(1).capitalize()
                if response_lang == 'es':
                     response_text = f"¡Hola {name_to_greet}!"
                else:
                     response_text = f"Hello {name_to_greet}!"
            # --- End of Handling for "say hello to [name]" ---

            # Existing handling for "greet [name]" directly after greet word
            elif re.match(r"(greet|hello|hi|hey)\s*([a-zA-Z]+)", user_input.lower()):
                 greet_name_only_match = re.match(r"(greet|hello|hi|hey)\s*([a-zA-Z]+)", user_input.lower())
                 if greet_name_only_match:
                     name = greet_name_only_match.group(2).capitalize()
                     if response_lang == 'es':
                         response_text = f"¡Hola {name}!"
                     else:
                         response_text = f"Hello {name}!"

            else:
                # Existing general greeting response, now using remembered name if available
                # This will be used for subsequent greetings after the initial one
                if response_lang == 'es':
                    response_text = f"¡Hola {user_name}! ¿En qué puedo ayudarte hoy?" if user_name else "¿Cómo puedo ayudarte?"
                elif response_lang == 'hi':
                     response_text = f"नमस्ते {user_name}! मैं आपकी कैसे मदद कर सकता हूँ?" if user_name else "नमस्ते! मैं आपकी कैसे मदद कर सकता हूँ?"
                elif response_lang == 'ur':
                     response_text = f"اسلام علیکم {user_name}! میں آپ کی کیسے مدد کر سکتا ہوں؟" if user_name else "اسلام علیکم! میں آپ کی کیسے مدد کر سکتا ہوں؟"
                elif response_lang == 'bn':
                     response_text = f"হ্যালো {user_name}! আমি আপনাকে কিভাবে সাহায্য করতে পারি?" if user_name else "হ্যালো! আমি আপনাকে কিভাবে সাহায্য করতে পারি?"
                elif response_lang == 'ja':
                     response_text = f"こんにちは {user_name}さん！今日はどのようにお手伝いできますか？" if user_name else "こんにちは！どのようにお手伝いできますか？"
                elif response_lang == 'ko':
                     response_text = f"안녕하세요 {user_name}님! 오늘은 어떻게 도와드릴까요?" if user_name else "안녕하세요! 어떻게 도와드릴까요?"
                else: # Default to English
                    response_text = f"Hello {user_name}, welcome back! How may I assist you today?" if user_name else "Hello, how may I assist you today?"


        elif intent == "ask for help":
            if response_lang == 'es':
                response_text = "Puedo ayudarte con llamadas de emergencia, chistes, traducciones, ubicación, clima y más."
            elif response_lang == 'hi':
                 response_text = "मैं आपातकालीन कॉल, चुटकुले, अनुवाद, स्थान, मौसम और बहुत कुछ में मदद कर सकता हूँ।"
            elif response_lang == 'ur':
                 response_text = "میں ہنگامی کالز، لطیفے، تراجم، مقام، موسم اور بہت کچھ میں مدد کر سکتا ہوں۔"
            elif response_lang == 'bn':
                 response_text = "আমি জরুরী কল, কৌতুক, অনুবাদ, অবস্থান, আবহাওয়া এবং আরও অনেক কিছুতে সাহায্য করতে পারি।"
            elif response_lang == 'ja':
                 response_text = "緊急通報、ジョーク、翻訳、場所、天気などをお手伝いできます。"
            elif response_lang == 'ko':
                 response_text = "긴급 통화、농담、번역、위치、날씨 등을 도와드릴 수 있습니다."
            else: # Default to English
                response_text = "I can help with emergency calls, jokes, translations, location, weather, and more."

        elif intent == "tell a joke":
            selected_joke = random.choice(jokes) # Select an English joke first

            # --- Handling: Check for requested language in the joke request ---
            # Look for "in [language name]"
            joke_lang_match = re.search(r"in\s+(hindi|spanish|urdu|bangla|bengali|japanese|korean)", user_input.lower()) # Added korean
            if joke_lang_match:
                requested_language = joke_lang_match.group(1).lower()
                # Map language names to language codes
                lang_code = {'hindi': 'hi', 'spanish': 'es', 'urdu': 'ur', 'bangla': 'bn', 'bengali': 'bn', 'japanese': 'ja', 'korean': 'ko'}.get(requested_language, 'en') # Added korean

                if lang_code != 'en': # If a non-English language was requested
                    speak(f"Okay, here's a joke in {requested_language}.") # Acknowledge the translation request
                    translate_and_speak(selected_joke, lang_code)
                    continue # Skip the regular speak below
                else: # If English was requested in the pattern
                     speak(selected_joke, 'en') # Tell the joke in English
                     continue # Skip the regular speak below


            # --- End of Handling ---

            # If no specific language was requested in the joke phrase,
            # proceed with the response based on the general response_lang
            if response_lang != 'en': # If the active language mode is not English
                 speak(f"Okay, here's a joke in {response_lang}.") # Acknowledge the translation
                 translate_and_speak(selected_joke, response_lang)
                 continue # Skip the regular speak below
            # If active_language is English, fall through to the regular speak
            else:
                pass # Fall through to the speak(response_text, response_lang) below

            # Fallback speak if no specific language was requested and active_language is English
            # or if translation failed.
            speak(selected_joke, response_lang)
            continue # Move to the next iteration after telling the joke


        elif intent == "joke feedback":
            if response_lang == 'es':
                response_text = "Uy, público difícil... Trabajaré en eso."
            elif response_lang == 'hi':
                 response_text = "ओह, मुश्किल दर्शक... मैं इस पर काम करूँगा।"
            elif response_lang == 'ur':
                 response_text = "اوہ، مشکل سامعین... میں اس پر کام کروں گا۔"
            elif response_lang == 'bn':
                 response_text = "ওহ, কঠিন শ্রোতা... আমি এটা নিয়ে কাজ করব।"
            elif response_lang == 'ja':
                 response_text = "うーん、難しい観客ですね... 改善します。"
            elif response_lang == 'ko':
                 response_text = "음, 어려운 관객이네요... 노력하겠습니다."
            else: # Default to English
                response_text = "Ouch, tough crowd... I'll work on that."

        elif intent == "thank you":
            if response_lang == 'es':
                response_text = "¡De nada!"
            elif response_lang == 'hi':
                 response_text = "आपका स्वागत है!"
            elif response_lang == 'ur':
                 response_text = "خوش آمدید!"
            elif response_lang == 'bn':
                 response_text = "আপনাকে স্বাগতম!"
            elif response_lang == 'ja':
                 response_text = "どういたしまして！"
            elif response_lang == 'ko':
                 response_text = "천만에요!"
            else: # Default to English
                response_text = "You're welcome!"

        elif intent == "introduce myself":
            # This intent is still useful if the user doesn't give their name initially
            name = extract_name(user_input) # Extract the name and update user_name
            if response_lang == 'es':
                response_text = f"¡Mucho gusto, {name}!" if name else "¡Mucho gusto!"
            elif response_lang == 'hi':
                 response_text = f"आपसे मिलकर अच्छा लगा, {name}!" if name else "आपसे मिलकर अच्छा लगा!"
            elif response_lang == 'ur':
                 response_text = f"آپ سے مل کر اچھا لگا، {name}!" if name else "آپ سے مل کر اچھا لگا!"
            elif response_lang == 'bn':
                 response_text = f"আপনার সাথে দেখা করে ভালো লাগলো, {name}!" if name else "আপনার সাথে দেখা করে ভালো লাগলো!"
            elif response_lang == 'ja':
                 response_text = f"{name}さん、はじめまして！" if name else "はじめまして！"
            elif response_lang == 'ko':
                 response_text = f"{name}님、만나서 반가워요!" if name else "만나서 반가워요!"
            else: # Default to English
                response_text = f"Nice to meet you, {name}!" if name else "Nice to meet you!"

        elif intent == "ask location":
            location = get_location() # Get the simulated location
            if response_lang == 'es':
                response_text = f"Actualmente estás en {location}."
            elif response_lang == 'hi':
                 response_text = f"आप अभी {location} में हैं।"
            elif response_lang == 'ur':
                 response_text = f"آپ اس وقت {location} میں ہیں۔"
            elif response_lang == 'bn':
                 response_text = f"আপনি বর্তমানে {location} এ আছেন।"
            elif response_lang == 'ja':
                 response_text = f"現在地は{location}です。"
            elif response_lang == 'ko':
                 response_text = f"현재 위치는 {location}입니다。"
            else: # Default to English
                response_text = f"You're currently in {location}."

        elif intent == "tell time":
            response_text = tell_time() # Get the current time (English format)
            # No translation for time format for simplicity, just prepend a phrase in the active language
            if response_lang == 'es':
                response_text = f"La hora actual es {response_text}."
            elif response_lang == 'hi':
                 response_text = f"अभी समय हुआ है {response_text}।"
            elif response_lang == 'ur':
                 response_text = f"اس وقت ہوا ہے {response_text}۔"
            elif response_lang == 'bn':
                 response_text = f"এখন সময় হলো {response_text}।"
            elif response_lang == 'ja':
                 response_text = f"現在の時刻は{response_text}です。"
            elif response_lang == 'ko':
                 response_text = f"현재 시간은 {response_text}입니다。"
            else: # Default to English
                response_text = f"The current time is {response_text}."

        elif intent == "ask weather":
            weather_info = get_weather() # Get the simulated weather
            if response_lang == 'es':
                # For simplicity, we'll just translate a generic phrase and append the English weather info
                response_text = f"El clima es: {weather_info}"
            elif response_lang == 'hi':
                 response_text = f"मौसम है: {weather_info}"
            elif response_lang == 'ur':
                 response_text = f"موسم ہے: {weather_info}"
            elif response_lang == 'bn':
                 response_text = f"আবহাওয়া হলো: {weather_info}"
            elif response_lang == 'ja':
                 response_text = f"天気は：{weather_info}"
            elif response_lang == 'ko':
                 response_text = f"날씨는: {weather_info}"
            else: # Default to English
                response_text = weather_info

        elif intent == "translate":
            # Handle translation requests based on target language keywords
            if "hindi" in user_input.lower():
                # Extract the phrase to translate
                phrase = re.sub(r"(say this in|translate this to|convert to)\s+hindi", "", user_input, flags=re.IGNORECASE).strip()
                if phrase:
                    translate_and_speak(phrase, "hi")
                else:
                    speak("कृपया अनुवाद के लिए वाक्यांश कहें।", active_language) # Ask for phrase in active language
                continue # Skip the regular speak
            elif "spanish" in user_input.lower():
                phrase = re.sub(r"(say this in|translate this to)\s+spanish", "", user_input, flags=re.IGNORECASE).strip()
                if phrase:
                    translate_and_speak(phrase, "es")
                else:
                    speak("Por favor, di la frase que quieres traducir al español.", active_language) # Ask for phrase in active language
                continue # Skip the regular speak
            elif "urdu" in user_input.lower():
                phrase = re.sub(r"(say this in|translate this to)\s+urdu", "", user_input, flags=re.IGNORECASE).strip()
                if phrase:
                    translate_and_speak(phrase, "ur")
                else:
                    speak("Please say the phrase you want to translate to Urdu.", active_language) # Ask for phrase in active language
                continue # Skip the regular speak
            elif "bangla" in user_input.lower() or "bengali" in user_input.lower():
                phrase = re.sub(r"(say this in|translate this to|convert to)\s+(bangla|bengali)", "", user_input, flags=re.IGNORECASE).strip()
                if phrase:
                    translate_and_speak(phrase, "bn")
                else:
                    speak("Please say the phrase you want to translate to Bangla.", active_language) # Ask for phrase in active language
                continue # Skip the regular speak
            elif "japanese" in user_input.lower(): # Added Japanese translation
                phrase = re.sub(r"(say this in|translate this to|convert to)\s+japanese", "", user_input, flags=re.IGNORECASE).strip()
                if phrase:
                    translate_and_speak(phrase, "ja")
                else:
                    speak("翻訳したいフレーズを言ってください。", active_language) # Ask for phrase in active language
                continue # Skip the regular speak
            elif "korean" in user_input.lower(): # Added Korean translation
                phrase = re.sub(r"(say this in|translate this to|convert to)\s+korean", "", user_input, flags=re.IGNORECASE).strip()
                if phrase:
                    translate_and_speak(phrase, "ko")
                else:
                    speak("번역하고 싶은 문구를 말해주세요。", active_language) # Ask for phrase in active language
                continue # Skip the regular speak
            else:
                speak("Sorry, I can only translate to Hindi, Spanish, Urdu, Bangla, Japanese, and Korean for now.", active_language) # Respond in active language
                continue # Skip the regular speak

        elif intent == "ask name":
            # Check if the user is asking Joey's name or their own name
            if any(phrase in user_input.lower() for phrase in ["what is your name", "who are you", "your name"]):
                 if response_lang == 'es':
                      response_text = "Soy Joey."
                 elif response_lang == 'hi':
                      response_text = "मैं जॉय हूँ।"
                 elif response_lang == 'ur':
                      response_text = "میں جوی ہوں۔"
                 elif response_lang == 'bn':
                      response_text = "আমি জোয়।"
                 elif response_lang == 'ja':
                      response_text = "私はジョイです。"
                 elif response_lang == 'ko':
                      response_text = "저는 조이입니다。"
                 else: # Default to English
                      response_text = "I am Joey."
            else: # Assume they are asking their own name
                if response_lang == 'es':
                    response_text = f"Tu nombre es {user_name}." if user_name else "Aún no sé tu nombre."
                elif response_lang == 'hi':
                     response_text = f"आपका नाम {user_name} है।" if user_name else "मुझे अभी तक आपका नाम नहीं पता।"
                elif response_lang == 'ur':
                     response_text = f"آپ کا نام {user_name} ہے۔" if user_name else "مجھے ابھی تک آپ کا نام نہیں معلوم۔"
                elif response_lang == 'bn':
                     response_text = f"আপনার নাম {user_name}।" if user_name else "আমি এখনো আপনার নাম জানি না।"
                elif response_lang == 'ja':
                     response_text = f"あなたの名前は{user_name}です。" if user_name else "まだあなたの名前を知りません。"
                elif response_lang == 'ko':
                     response_text = f"당신의 이름은 {user_name}입니다。" if user_name else "아직 당신의 이름을 몰라요。"
                else: # Default to English
                    response_text = f"Your name is {user_name}." if user_name else "I don't know your name yet."

        elif intent == "ask origin":
            if response_lang == 'es':
                 response_text = "Anni me hizo."
            elif response_lang == 'hi':
                 response_text = "अन्नी ने मुझे बनाया।"
            elif response_lang == 'ur':
                 response_text = "انی نے مجھے بنایا۔"
            elif response_lang == 'bn':
                 response_text = "আন্নি আমাকে তৈরি করেছে।"
            elif response_lang == 'ja':
                 response_text = "アンニが私を作りました。"
            elif response_lang == 'ko':
                 response_text = "안니가 저를 만들었습니다。"
            else: # Default to English
                 response_text = "Anni did."

        elif intent == "ask appearance":
            if response_lang == 'es':
                 response_text = "No tengo cabello ni apariencia física, soy un programa de voz."
            elif response_lang == 'hi':
                 response_text = "मेरे बाल या कोई शारीरिक रूप नहीं है， मैं एक आवाज सहायक हूँ。"
            elif response_lang == 'ur':
                 response_text = "میرے بال یا کوئی جسمانی شکل نہیں ہے، میں ایک آواز معاون ہوں۔"
            elif response_lang == 'bn':
                 response_text = "আমার চুল বা শারীরিক চেহারা নেই, আমি একটি ভয়েس অ্যাসისტ্যান্ট।"
            elif response_lang == 'ja':
                 response_text = "髪や物理的な外見はありません、私は音声アシスタントです。"
            elif response_lang == 'ko':
                 response_text = "머리카락이나 물리적인 외모는 없어요， 저는 음성 비서입니다。"
            else: # Default to English
                 response_text = "I don't have hair or a physical appearance, I'm a voice assistant."

        elif intent == "ask age":
             if response_lang == 'es':
                  response_text = "No tengo edad en el sentido humano, soy un programa de computadora."
             elif response_lang == 'hi':
                  response_text = "मानवीय अर्थ में मेरी कोई उम्र नहीं है, मैं एक कंप्यूटर प्रोग्राम हूँ।"
             elif response_lang == 'ur':
                  response_text = "انسانی معنوں میں میری کوئی عمر نہیں ہے، میں ایک کمپیوٹر پروگرام ہوں۔"
             elif response_lang == 'bn':
                  response_text = "মানুষের অর্থে আমার কোনো বয়স নেই, আমি একটি কম্পিউটার প্রোগ্রাম।"
             elif response_lang == 'ja':
                  response_text = "人間のような年齢はありません、私はコンピュータプログラムです。"
             elif response_lang == 'ko':
                  response_text = "인간적인 의미의 나이는 없어요， 저는 컴퓨터 프로그램입니다。"
             else: # Default to English
                  response_text = "I don't have an age in the human sense, I'm a computer program."

        elif intent == "ask vehicle info":
             # Simulate getting dummy vehicle info
             car_make = "Honda"
             car_model = "City"
             fuel_level = random.randint(10, 90) # Random fuel level
             tire_pressure = f"{random.randint(30, 35)} PSI" # Random tire pressure
             mileage = f"{random.randint(10000, 100000):,} km" # Random mileage

             # Create the base English response text
             base_english_response = f"Here is your car information: Make {car_make}, Model {car_model}. Fuel level: {fuel_level} percent. Tire pressure: {tire_pressure}. Mileage: {mileage}."

             # Check for requested language in the vehicle info request
             vehicle_info_lang_match = re.search(r"in\s+(hindi|spanish|urdu|bangla|bengali|japanese|korean)", user_input.lower())
             if vehicle_info_lang_match:
                  requested_language = vehicle_info_lang_match.group(1).lower()
                  # Map language names to language codes
                  lang_code = {'hindi': 'hi', 'spanish': 'es', 'urdu': 'ur', 'bangla': 'bn', 'bengali': 'bn', 'japanese': 'ja', 'korean': 'ko'}.get(requested_language, 'en')

                  if lang_code != 'en': # If a non-English language was requested
                       speak(f"Okay, here is your car information in {requested_language}.") # Acknowledge the translation request
                       translate_and_speak(base_english_response, lang_code)
                       continue # Skip the regular speak below
                  # If English was requested in the pattern, it will fall through to the default English response

             # If no specific language was requested in the phrase, use the active language mode
             if response_lang == 'es':
                  response_text = f"La información de tu coche es: Marca {car_make}, Modelo {car_model}. Nivel de combustible: {fuel_level} por ciento. Presión de los neumáticos: {tire_pressure}. Kilometraje: {mileage}."
             elif response_lang == 'hi':
                  response_text = f"आपकी कार की जानकारी यह है: मेक {car_make}, मॉडल {car_model}। ईंधन स्तर: {fuel_level} प्रतिशत। टायर का दबाव: {tire_pressure}। माइलेज: {mileage}।"
             elif response_lang == 'ur':
                  response_text = f"آپ کی کار کی معلومات یہ ہیں: میک {car_make}، ماڈل {car_model}۔ ایندھن کی سطح: {fuel_level} فیصد۔ ٹائر کا دباؤ: {tire_pressure}۔ مائلیج: {mileage}۔"
             elif response_lang == 'bn':
                  response_text = f"আপনার গাড়ির তথ্য হলো: মেক {car_make}, মডেল {car_model}। জ্বালানির স্তর: {fuel_level} শতাংশ। টায়ারের চাপ: {tire_pressure}। মাইলেজ: {mileage}।"
             elif response_lang == 'ja':
                  response_text = f"あなたの車の情報は次のとおりです：メーカー {car_make}、モデル {car_model}。燃料レベル：{fuel_level}パーセント。タイヤ空気圧：{tire_pressure}。走行距離：{mileage}。"
             elif response_lang == 'ko':
                  response_text = f"차량 정보입니다: 제조사 {car_make}, 모델 {car_model}. 연료 잔량: {fuel_level} 퍼센트. 타이어 공기압: {tire_pressure}. 주행 거리: {mileage}."
             else: # Default to English
                  response_text = base_english_response # Use the base English response

             # Speak the generated response text if it's not empty (this handles the case where no specific language was requested in the phrase)
             if response_text:
                 speak(response_text, response_lang)
             continue # Continue to the next loop iteration

        elif intent == "tell riddle":
             selected_riddle = random.choice(riddles)
             riddle_text = selected_riddle["riddle"]
             riddle_answer = selected_riddle["answer"]

             # Check for requested language in the riddle request
             riddle_lang_match = re.search(r"in\s+(hindi|spanish|urdu|bangla|bengali|japanese|korean)", user_input.lower())
             if riddle_lang_match:
                  requested_language = riddle_lang_match.group(1).lower()
                  lang_code = {'hindi': 'hi', 'spanish': 'es', 'urdu': 'ur', 'bangla': 'bn', 'bengali': 'bn', 'japanese': 'ja', 'korean': 'ko'}.get(requested_language, 'en')

                  if lang_code != 'en':
                       speak(f"Okay, here is a riddle in {requested_language}.")
                       translate_and_speak(riddle_text, lang_code)
                       time.sleep(5) # Wait for 5 seconds for the user to answer
                       speak("The answer is...")
                       translate_and_speak(riddle_answer, lang_code)
                       continue # Skip the regular speak below
                  # If English was requested in the pattern, it will fall through to the default English response

             # If no specific language was requested in the phrase, use the active language mode
             if response_lang != 'en':
                  speak("Here is a riddle for you.")
                  translate_and_speak(riddle_text, response_lang)
                  time.sleep(5) # Wait for 5 seconds for the user to answer
                  speak("The answer is...")
                  translate_and_speak(riddle_answer, response_lang)
             else:
                  speak("Here is a riddle for you.")
                  speak(riddle_text, response_lang)
                  time.sleep(5) # Wait for 5 seconds for the user to answer
                  speak("The answer is...")
                  speak(riddle_answer, response_lang)

             continue # Move to the next iteration after telling the riddle

        elif intent == "tell fact":
             selected_fact = random.choice(facts)

             # Check for requested language in the fact request
             fact_lang_match = re.search(r"in\s+(hindi|spanish|urdu|bangla|bengali|japanese|korean)", user_input.lower())
             if fact_lang_match:
                  requested_language = fact_lang_match.group(1).lower()
                  lang_code = {'hindi': 'hi', 'spanish': 'es', 'urdu': 'ur', 'bangla': 'bn', 'bengali': 'bn', 'japanese': 'ja', 'korean': 'ko'}.get(requested_language, 'en')

                  if lang_code != 'en':
                       speak(f"Okay, here is a fact in {requested_language}.")
                       translate_and_speak(selected_fact, lang_code)
                       continue # Skip the regular speak below
                  # If English was requested in the pattern, it will fall through to the default English response

             # If no specific language was requested in the phrase, use the active language mode
             if response_lang != 'en':
                  speak("Here is a random fact for you.")
                  translate_and_speak(selected_fact, response_lang)
             else:
                  speak("Here is a random fact for you.")
                  speak(selected_fact, response_lang)
             continue # Move to the next iteration after telling the fact


        elif "stop" in user_input.lower() or "exit" in user_input.lower() or intent == "stop or exit":
             # Handle stop or exit commands
            if active_language == 'es':
                response_text = "¡Adiós! Cuídate."
            elif active_language == 'hi':
                 response_text = "अलविदा! अपना ख्याल रखना。"
            elif active_language == 'ur':
                 response_text = "خدا حافظ! اپنا خیال رکھنا۔"
            elif active_language == 'bn':
                 response_text = "বিদায়! নিজের যত্ন নিয়ো。"
            elif active_language == 'ja':
                 response_text = "さようなら！お気をつけて。"
            elif active_language == 'ko':
                 response_text = "안녕히 가세요! 조심하세요。"
            else: # Default to English
                response_text = "Goodbye! Take care."
            speak(response_text, active_language) # Speak the goodbye message in the active language
            break # Exit the main loop
        elif intent == "emergency call":
            # Handle emergency calls (already handled by handle_distress_signal, but included for completeness)
            handle_distress_signal(user_input, score, user_lang)
            continue # Skip the regular speak
        else:
            # Default response for unmatched intents
            if response_lang == 'es':
                response_text = "Lo siento, no entendí eso."
            elif response_lang == 'hi':
                 response_text = "क्षमा करें, मुझे वह समझ नहीं आया।"
            elif response_lang == 'ur':
                 response_text = "معاف کیجئے گا، مجھے وہ سمجھ نہیں آیا۔"
            elif response_lang == 'bn':
                 response_text = "দুঃখিত, আমি এটা বুঝতে পারিনি。"
            elif response_lang == 'ja':
                 response_text = "すみません、理解できませんでした。"
            elif response_lang == 'ko':
                 response_text = "죄송해요， 이해하지 못했어요。"
            else: # Default to English
                response_text = "Sorry, I didn't understand that."

        # Speak the generated response text if it's not empty
        if response_text:
            speak(response_text, response_lang)

        # Perform background driving checks after handling user input
        check_and_warn_speeding(previous_user_lang)
        check_and_warn_traffic_light(previous_user_lang)
        time.sleep(1) # Small delay to avoid busy-waiting

if __name__ == "__main__":
    # Entry point of the script
    main()
