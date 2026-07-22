"""
Campus Buddy - A simple NLP-based chatbot
==========================================
Built as a mini-project to demonstrate NLP concepts:
- Text preprocessing (tokenization, lowercasing)
- Intent matching using TF-IDF vectorization
- Cosine similarity for finding the best matching response

Author: <your name here>
"""

import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# 1. Knowledge base (intents)
# -----------------------------
# Each intent has a list of example phrases (patterns) and possible responses.
intents = {
    "greeting": {
        "patterns": ["hi", "hello", "hey", "good morning", "good evening", "is anyone there"],
        "responses": ["Hello! I'm Campus Buddy 🎓. How can I help you today?",
                      "Hi there! Ask me about courses, exams, library, or internships."]
    },
    "courses": {
        "patterns": ["what courses are offered", "tell me about courses", "list of courses",
                     "which programs do you have", "course details"],
        "responses": ["We offer B.Tech, B.Sc, BCA, MCA, and Diploma programs across CS, ECE, and Mechanical streams."]
    },
    "exam": {
        "patterns": ["when are the exams", "exam schedule", "exam date sheet", "when is the semester exam"],
        "responses": ["Semester exams usually start in the last week of the month. Check the notice board or student portal for the exact date sheet."]
    },
    "library": {
        "patterns": ["library timings", "when does the library open", "library hours", "is the library open"],
        "responses": ["The library is open from 8 AM to 8 PM on weekdays, and 9 AM to 4 PM on weekends."]
    },
    "library_closing": {
        "patterns": ["library closing time", "when does library close", "library close early",
                     "special library hours", "library exam time hours"],
        "responses": ["The library closes at 6 PM on Saturdays, and stays shut on public holidays. "
                      "During exam week, it stays open till 10 PM as a special extended hour."]
    },
    "timetable": {
        "patterns": ["class schedule", "timetable", "what time is my class", "class timing",
                     "when is my next class", "weekly schedule", "full timetable"],
        "responses": ["IV AIML B (Batch 1) Weekly Timetable:\n"
                      "Mon: A-C-D-E, then Placement Training\n"
                      "Tue: B-D-E-F, then Placement Training\n"
                      "Wed: C-F-B, then Golden Hour, then Placement Training\n"
                      "Thu: F-B-E-A, then Placement Training\n"
                      "Fri: D-A-C, then Placement Training\n"
                      "(A=Behavioral Psychology, B=Business Intelligence & Analytics, C=IoT Concepts, "
                      "D=Deep Learning Techniques, E=Report Writing, F=Renewable Energy Sources)"]
    },
    "monday_schedule": {
        "patterns": ["monday timetable", "monday classes", "what classes on monday", "monday schedule"],
        "responses": ["Monday: 9.00-9.50 Behavioral Psychology (A), 9.55-10.45 IoT Concepts (C), "
                      "10.50-11.40 Deep Learning (D), 11.45-12.35 Report Writing (E), "
                      "then Placement Training from 12.40 till 5.05."]
    },
    "tuesday_schedule": {
        "patterns": ["tuesday timetable", "tuesday classes", "what classes on tuesday", "tuesday schedule"],
        "responses": ["Tuesday: 9.00-9.50 Business Intelligence (B), 9.55-10.45 Deep Learning (D), "
                      "10.50-11.40 Report Writing (E), 11.45-12.35 Renewable Energy (F), "
                      "then Placement Training from 12.40 till 5.05."]
    },
    "wednesday_schedule": {
        "patterns": ["wednesday timetable", "wednesday classes", "what classes on wednesday", "wednesday schedule"],
        "responses": ["Wednesday: 9.00-9.50 IoT Concepts (C), 9.55-10.45 Renewable Energy (F), "
                      "10.50-11.40 Business Intelligence (B), then Golden Hour from 11.45-1.30, "
                      "then Placement Training from 1.30 till 5.05."]
    },
    "thursday_schedule": {
        "patterns": ["thursday timetable", "thursday classes", "what classes on thursday", "thursday schedule"],
        "responses": ["Thursday: 9.00-9.50 Renewable Energy (F), 9.55-10.45 Business Intelligence (B), "
                      "10.50-11.40 Report Writing (E), 11.45-12.35 Behavioral Psychology (A), "
                      "then Placement Training from 12.40 till 5.05."]
    },
    "friday_schedule": {
        "patterns": ["friday timetable", "friday classes", "what classes on friday", "friday schedule"],
        "responses": ["Friday: 9.00-9.50 Deep Learning (D), 9.55-10.45 Behavioral Psychology (A), "
                      "10.50-11.40 IoT Concepts (C), then Placement Training from 11.45 till 5.05."]
    },
    "golden_hour": {
        "patterns": ["what is golden hour", "golden hour timing", "when is golden hour"],
        "responses": ["Golden Hour is a special slot on Wednesday from 11.45 AM to 1.30 PM, "
                      "usually reserved for open activities, mentoring, or flexible academic use."]
    },
    "placement_training": {
        "patterns": ["placement training timing", "when is placement training", "placement training schedule"],
        "responses": ["Placement Training runs every weekday in the afternoon (from around 12.40/1.30 PM "
                      "until 5.05 PM depending on the day), covering periods 6 through 9."]
    },
    "faculty": {
        "patterns": ["who teaches iot", "deep learning faculty", "who teaches deep learning",
                     "business intelligence faculty", "who is the faculty for report writing"],
        "responses": ["Faculty list: A - Behavioral Psychology (Ms. Harshavarthini V), "
                      "B - Business Intelligence & Analytics (Dr. Venkatesh Guru), "
                      "C - IoT Concepts (Dr. Siron Anita Susan), "
                      "D - Deep Learning Techniques (Dr. Thurai Pandian), "
                      "E - Report Writing (Dr. B. Padma Priya), "
                      "F - Renewable Energy Sources (Dr. M. Sridharan)."]
    },
    "internship": {
        "patterns": ["internship opportunities", "tell me about internships", "how to apply for internship",
                     "internship guidance"],
        "responses": ["You can check AICTE's internship portal, IBM SkillsBuild, or your college placement cell for the latest internship openings."]
    },
    "contact": {
        "patterns": ["how do i contact college", "college phone number", "college email", "admin office contact"],
        "responses": ["You can reach the college admin office at admin@college.edu or call the front desk during working hours."]
    },
    "thanks": {
        "patterns": ["thank you", "thanks", "that helped", "appreciate it"],
        "responses": ["You're welcome! Happy to help 😊", "Anytime! Good luck with your studies."]
    },
    "goodbye": {
        "patterns": ["bye", "goodbye", "see you", "exit", "quit"],
        "responses": ["Goodbye! Have a great day.", "See you next time!"]
    },
    "unknown": {
        # Example off-topic phrases so the model learns what NOT to match confidently.
        "patterns": ["tell me a joke", "what's the weather", "sing a song", "who are you really",
                     "what is the meaning of life", "can you help me with math homework"],
        "responses": ["Sorry, I didn't quite understand that. Could you rephrase your question?",
                      "I'm only trained to help with campus-related questions right now!"]
    }
}

# -----------------------------
# 2. Prepare training data
# -----------------------------
all_patterns = []
pattern_to_intent = []

for intent_name, intent_data in intents.items():
    for pattern in intent_data["patterns"]:
        all_patterns.append(pattern)
        pattern_to_intent.append(intent_name)

# TF-IDF vectorizer converts text into numeric vectors based on word importance
vectorizer = TfidfVectorizer()
pattern_vectors = vectorizer.fit_transform(all_patterns)

# -----------------------------
# 3. Core matching function
# -----------------------------
def get_response(user_input, threshold=0.3):
    """
    Finds the most similar known pattern to the user's input using
    cosine similarity on TF-IDF vectors, and returns a matching response.
    """
    user_vector = vectorizer.transform([user_input.lower()])
    similarities = cosine_similarity(user_vector, pattern_vectors)
    best_match_index = similarities.argmax()
    best_score = similarities[0, best_match_index]

    if best_score < threshold:
        return "Sorry, I didn't quite understand that. Could you rephrase your question?"

    matched_intent = pattern_to_intent[best_match_index]
    return random.choice(intents[matched_intent]["responses"])


# -----------------------------
# 4. Chat loop
# -----------------------------
def run_chatbot():
    print("=" * 50)
    print(" Campus Buddy Chatbot - type 'quit' to exit")
    print("=" * 50)

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Bot:", get_response(user_input))
            break
        response = get_response(user_input)
        print("Bot:", response)


if __name__ == "__main__":
    run_chatbot()
