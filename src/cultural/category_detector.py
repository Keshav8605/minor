import re

CATEGORY_KEYWORDS = {
    "family": ["mom", "dad", "parents", "sharma", "relatives", "mummy", "papa", "bhai", "mother", "father"],
    "education": ["school", "study", "marks", "teacher", "principal", "engineering"],
    "college": ["hostel", "placement", "attendance", "cgpa", "assignment", "professor", "college"],
    "JEE/exams": ["jee", "neet", "kota", "coaching", "exam", "upsc", "result"],
    "cricket": ["cricket", "dhoni", "kohli", "ipl", "match", "rohit", "rcb", "csk"],
    "Bollywood": ["movie", "actor", "salman", "srk", "akshay", "hera pheri", "bollywood", "film"],
    "food": ["chai", "momo", "pani puri", "biryani", "paneer", "food"],
    "festivals": ["diwali", "holi", "rakhi", "eid", "christmas", "puja", "cleaning", "festival"],
    "marriage/wedding": ["wedding", "shaadi", "marriage", "arrange", "bride", "groom"],
    "workplace": ["boss", "salary", "appraisal", "corporate", "office", "manager"],
    "relationships": ["gf", "bf", "dating", "breakup", "crush", "single"],
    "social_norms": ["log kya kahenge", "society", "jugaad", "bargain"],
    "daily_life": ["traffic", "train", "auto", "middle class"]
}

def detect_categories(text: str) -> list:
    """
    Scans text for cultural keywords.
    Returns a list of matching categories.
    """
    text = text.lower()
    detected = set()
    
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if re.search(r'(?:^|\W)' + re.escape(kw) + r'(?:$|\W)', text):
                detected.add(category)
                break
                
    if not detected:
        detected.add("none")
        
    return list(detected)
