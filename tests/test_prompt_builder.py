import unittest
from src.vlm.prompts import (
    build_humor_analysis_prompt,
    build_cultural_analysis_prompt,
    build_multi_image_prompt
)

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

    def test_build_cultural_prompt(self):
        messages = build_cultural_analysis_prompt(
            "dummy/path.jpg",
            ocr_text="Sharma ji ka beta",
            retrieved_context="Indian family comparison trope"
        )
        self.assertEqual(len(messages), 2)
        user_text = messages[1]["content"][1]["text"]
        self.assertIn("Sharma ji ka beta", user_text)
        self.assertIn("Indian family comparison trope", user_text)
        self.assertIn("cultural_category", user_text)
        self.assertIn("cultural_dependency", user_text)

    def test_build_multi_image_prompt(self):
        paths = ["dummy/img1.jpg", "dummy/img2.jpg"]
        messages = build_multi_image_prompt(paths, mode="general")
        self.assertEqual(len(messages), 2)
        user_content = messages[1]["content"]
        # Should have 2 images + 1 text prompt = 3 items
        self.assertEqual(len(user_content), 3)
        self.assertEqual(user_content[0]["type"], "image")
        self.assertEqual(user_content[1]["type"], "image")
        self.assertEqual(user_content[2]["type"], "text")
        self.assertIn("Analyze the above 2 meme images together", user_content[2]["text"])

