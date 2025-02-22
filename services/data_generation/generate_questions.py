import os
import json
import requests
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
import sys

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from services.data_generation.translate_nouns import translate_nouns

load_dotenv()

class QuestionGenerator:
    def __init__(self, target_lang: str):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.target_lang = target_lang
        self.translate_api_url = 'http://127.0.0.1:5000/translate'
        self.deepseek_api_url = "https://api.deepseek.com/v1/chat/completions"
        
        # Ensure data directory exists
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)

    def translate_text(self, text: str, source_lang: str = "en") -> str:
        """Translate text using the translation API"""
        try:
            response = requests.post(
                self.translate_api_url,
                json={
                    "q": text,
                    "source": source_lang,
                    "target": self.target_lang
                },
                headers={"Content-Type": "application/json"}
            )
            
            if response.ok:
                return response.json()['translatedText']
            return text
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def generate_questions_for_noun(self, noun: str) -> List[str]:
        """Generate unique questions about a noun using Deepseek API"""
        please_answer = self.translate_text("Please answer in English")
        
        prompt = f"""Generate 10 unique questions in {self.target_lang} about '{noun}'.
        Make each question different and interesting.
        Add "{please_answer}" at the end of each question.
        Return just the questions, one per line."""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.9
        }
        
        try:
            response = requests.post(
                self.deepseek_api_url,
                headers=headers,
                json=data
            )
            
            questions = response.json()['choices'][0]['message']['content'].split('\n')
            return [q.strip() for q in questions if q.strip()]
            
        except Exception as e:
            print(f"Deepseek API error for noun '{noun}': {e}")
            return []

    def process_nouns_and_generate_questions(self, num_questions_per_noun: int = 10):
        """Main process to generate questions for translated nouns"""
        # First, ensure nouns are translated
        translate_nouns(self.target_lang)
        
        # Read translated nouns
        translations_file = self.data_dir / f"nouns_{self.target_lang}.txt"
        if not translations_file.exists():
            raise FileNotFoundError(f"Translations file not found: {translations_file}")
        
        # Read noun translations
        noun_pairs = {}
        with open(translations_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '\t' in line:
                    en_noun, translated_noun = line.strip().split('\t')
                    noun_pairs[en_noun] = translated_noun

        # Generate questions for each noun
        all_questions = {}
        for en_noun, translated_noun in noun_pairs.items():
            print(f"Generating questions for noun: {translated_noun}")
            questions = self.generate_questions_for_noun(translated_noun)
            
            # Take only the requested number of questions
            questions = questions[:num_questions_per_noun]
            
            if questions:
                all_questions[translated_noun] = {
                    "english_noun": en_noun,
                    "translated_noun": translated_noun,
                    "questions": questions
                }

        # Save results
        output_file = self.data_dir / f"questions_{self.target_lang}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_questions, f, ensure_ascii=False, indent=2)
        
        print(f"Generated questions saved to {output_file}")
        return all_questions

def generate_questions(target_lang: str, questions_per_noun: int = 10):
    """Convenience function to generate questions"""
    generator = QuestionGenerator(target_lang)
    return generator.process_nouns_and_generate_questions(questions_per_noun)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_questions.py <target_language> <questions_per_noun>")
        sys.exit(1)
    
    target_lang = sys.argv[1]
    questions_per_noun = int(sys.argv[2])
    generate_questions(target_lang, questions_per_noun) 