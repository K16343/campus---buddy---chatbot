# Campus Buddy Chatbot 🎓

A simple NLP-based chatbot built in Python that answers common college-related 
questions — courses, exams, library hours, class timetable, and internships.

## How it works
- User input is converted into a TF-IDF vector (weighs important words)
- Compared against stored example questions using cosine similarity
- Returns the closest matching answer, or a fallback if nothing matches well

## Tech used
- Python
- scikit-learn (TfidfVectorizer, cosine_similarity)

## How to run
```bash
pip install scikit-learn
python3 campus_buddy_chatbot.py
```

## Example
## Future improvements
- Add voice input
- Deploy as a web app using Flask
- Connect to a real college database
