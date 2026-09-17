import re

CATEGORY_KEYWORDS = {
    "family": ["mom", "dad", "parents", "sharma", "relatives", "mummy", "papa", "bhai", "mother", "father", "behen", "didi", "chacha", "bhabhi", "rishtedar", "baap", "माँ", "पापा", "भाई", "बहन"],
    "education": ["school", "study", "marks", "teacher", "principal", "engineering", "padhai", "padhai-likhai", "homework", "tuition", "maths", "science", "पढ़ाई"],
    "college": ["hostel", "placement", "attendance", "cgpa", "assignment", "professor", "college", "canteen", "sem", "semester", "backlog", "kt", "ragging", "हॉस्टल"],
    "JEE/exams": ["jee", "neet", "kota", "coaching", "exam", "upsc", "result", "topper", "rank", "cutoff", "paper leak", "बोर्ड", "परीक्षा"],
    "cricket": ["cricket", "dhoni", "kohli", "ipl", "match", "rohit", "rcb", "csk", "thala", "king kohli", "bcci", "sixer", "wicket", "क्रिकेट"],
    "Bollywood": ["movie", "actor", "salman", "srk", "akshay", "hera pheri", "bollywood", "film", "dialogue", "baburao", "cinema", "फिल्म", "बॉलीवुड"],
    "food": ["chai", "momo", "pani puri", "biryani", "paneer", "food", "samosa", "golgappa", "maggi", "mithai", "chole bhature", "चाय", "समोसा"],
    "festivals": ["diwali", "holi", "rakhi", "eid", "christmas", "puja", "cleaning", "festival", "diya", "patakhe", "rangoli", "दिवाली", "होली"],
    "marriage/wedding": ["wedding", "shaadi", "marriage", "arrange", "bride", "groom", "rishta", "dulha", "dulhan", "barat", "mandap", "kanyadan", "शादी"],
    "workplace": ["boss", "salary", "appraisal", "corporate", "office", "manager", "wfh", "client", "hr", "cubicle", "overtime", "नौकरी", "ऑफिस"],
    "relationships": ["gf", "bf", "dating", "breakup", "crush", "single", "nibba", "nibbi", "ex", "proposal", "relationship", "प्यार"],
    "social_norms": ["log kya kahenge", "society", "jugaad", "bargain", "sanskar", "sharma ji", "padosi", "aunty", "mohalle", "जुगाड़"],
    "religion": ["mandir", "masjid", "pooja", "puja", "vrat", "prasad", "pandit", "bhagwan", "temple", "ramayan", "mahabharat", "मंदिर", "पूजा"],
    "regional_culture": ["delhi", "mumbai", "bangalore", "bengaluru", "bihar", "punjab", "punjabi", "gujju", "gujarat", "south indian", "local train", "noida", "gurgaon"],
    "daily_life": ["traffic", "train", "auto", "middle class", "bijli", "ration", "rickshaw", "metro", "chai ki tapri", "किराया"]
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
