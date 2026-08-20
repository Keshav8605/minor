import unittest
from src.vlm.prompts import build_humor_analysis_prompt

class TestPromptBuilder(unittest.TestCase):
    def test_build_prompt(self):
        messages = build_humor_analysis_prompt("dummy/path.jpg")
        
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        
        self.assertIn("JSON", messages[0]["content"][0]["text"])
        
        user_content = messages[1]["content"]
        self.assertEqual(user_content[0]["type"], "image")
        self.assertEqual(user_content[0]["image"], "file://dummy/path.jpg")
        self.assertEqual(user_content[1]["type"], "text")
        self.assertIn("Analyze the following meme", user_content[1]["text"])
