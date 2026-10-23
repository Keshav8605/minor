"""
Unit tests and regression checks for refined _CLASSIFICATION_RULES in src/vlm/prompts.py.
"""

import unittest
from src.vlm.prompts import _CLASSIFICATION_RULES, build_humor_analysis_prompt, build_cultural_analysis_prompt


class TestPromptClassificationRules(unittest.TestCase):
    def test_satirical_hypocrisy_included_in_comedic_mechanisms(self):
        """Rule B must recognize intentional satirical hypocrisy / self-contradictory personas."""
        self.assertIn("satirical hypocrisy", _CLASSIFICATION_RULES.lower())
        self.assertIn("self-contradictory personas", _CLASSIFICATION_RULES.lower())

    def test_celebrity_caricature_parody_distinguished(self):
        """Rule B/C must distinguish caricature parody merchandise from sincere personal greetings."""
        self.assertIn("celebrity caricature", _CLASSIFICATION_RULES.lower())
        self.assertIn("parody merchandise", _CLASSIFICATION_RULES.lower())
        self.assertIn("genuine personal, religious, or cultural greetings", _CLASSIFICATION_RULES.lower())

    def test_comic_format_defamatory_attacks_clarified(self):
        """Rule C/D must state that serious criminal allegations / defamatory attacks in webcomic formats are NOT humor."""
        self.assertIn("defamatory accusations", _CLASSIFICATION_RULES.lower())
        self.assertIn("comic, or webcomic", _CLASSIFICATION_RULES.lower())
        self.assertIn("serious criminal allegations", _CLASSIFICATION_RULES.lower())

    def test_conventional_punchline_not_strictly_required(self):
        """Rule B must not strictly require a conventional punchline if situational/visual/parody comedy is evident."""
        self.assertIn("punchline is not strictly required", _CLASSIFICATION_RULES.lower())

    def test_cultural_relevance_alone_not_humor(self):
        """Preserves research integrity principle: cultural relevance does NOT equal humor."""
        self.assertIn("cultural reference alone", _CLASSIFICATION_RULES.lower())
        self.assertIn("does not establish humor", _CLASSIFICATION_RULES.lower())

    def test_prompt_builders_incorporate_classification_rules(self):
        """Verifies that both prompt builders include _CLASSIFICATION_RULES in user prompt text."""
        general_prompt = build_humor_analysis_prompt("test.jpg")
        cultural_prompt = build_cultural_analysis_prompt("test.jpg", "ocr", "context")

        gen_user_text = general_prompt[1]["content"][1]["text"]
        cult_user_text = cultural_prompt[1]["content"][1]["text"]

        self.assertIn("CLASSIFICATION RULES:", gen_user_text)
        self.assertIn("CLASSIFICATION RULES:", cult_user_text)
        self.assertIn("satirical hypocrisy", gen_user_text.lower())
        self.assertIn("satirical hypocrisy", cult_user_text.lower())

    def test_political_or_celebrity_parody_greeting_cards_distinguished(self):
        """Rule B/C must distinguish political/celebrity parody greeting cards from genuine personal greetings."""
        self.assertIn("political or celebrity parody greeting cards", _CLASSIFICATION_RULES.lower())
        self.assertIn("genuine personal greetings", _CLASSIFICATION_RULES.lower())

    def test_situational_irony_public_announcements_vs_romantic_panic(self):
        """Rule B must recognize situational irony between serious announcements and melodramatic romance panic."""
        self.assertIn("situational irony", _CLASSIFICATION_RULES.lower())
        self.assertIn("serious public announcements", _CLASSIFICATION_RULES.lower())
        self.assertIn("teenage romance panic", _CLASSIFICATION_RULES.lower())

    def test_heinous_crime_smear_distinguished_from_dark_humor(self):
        """Rule C/D must distinguish unevidenced heinous crime smears against platforms/entities from genuine dark humor."""
        self.assertIn("heinous crimes", _CLASSIFICATION_RULES.lower())
        self.assertIn("character assassination or platform warfare", _CLASSIFICATION_RULES.lower())
        self.assertIn("does not constitute dark humor", _CLASSIFICATION_RULES.lower())

    def test_balanced_symmetric_classification_system_prompt(self):
        """System prompts must evaluate evidence symmetrically without arbitrary prior bias."""
        self.assertIn("balanced classification", _CLASSIFICATION_RULES.lower())
        self.assertIn("symmetrically", _CLASSIFICATION_RULES.lower())

        general_prompt = build_humor_analysis_prompt("test.jpg")
        cultural_prompt = build_cultural_analysis_prompt("test.jpg", "ocr", "context")

        gen_sys_text = general_prompt[0]["content"][0]["text"]
        cult_sys_text = cultural_prompt[0]["content"][0]["text"]

        self.assertIn("symmetrically", gen_sys_text.lower())
        self.assertIn("without prior class bias", gen_sys_text.lower())
        self.assertNotIn("most content you analyze will not be humorous", gen_sys_text.lower())
        self.assertNotIn("most content you analyze will not be humorous", cult_sys_text.lower())

    def test_fictional_dark_comedy_explicitly_protected(self):
        """Rule B must protect legitimate fictional dark comedy and morbid absurdity."""
        self.assertIn("legitimate dark comedy", _CLASSIFICATION_RULES.lower())
        self.assertIn("morbid absurdity", _CLASSIFICATION_RULES.lower())
        self.assertIn("fictional characters", _CLASSIFICATION_RULES.lower())

    def test_real_world_criminal_smear_boundary_defined(self):
        """Rule C/D must define real-world criminal accusations on comic templates as non-humor smears."""
        self.assertIn("real-world criminal accusations on comic templates", _CLASSIFICATION_RULES.lower())
        self.assertIn("identifiable real-world", _CLASSIFICATION_RULES.lower())
        self.assertIn("contrast with legitimate dark humor", _CLASSIFICATION_RULES.lower())

    def test_decision_test_includes_criminal_smear_check(self):
        """Rule E must include an explicit check against heinous crime accusations on real-world entities."""
        self.assertIn("heinous crime against an identifiable real-world platform", _CLASSIFICATION_RULES.lower())
        self.assertIn("if yes, set humorous=false", _CLASSIFICATION_RULES.lower())
