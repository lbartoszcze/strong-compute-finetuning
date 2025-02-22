import os
import json
import requests
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
import sys
import time

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from services.data_generation.translate_nouns import translate_nouns

load_dotenv()

class QuestionGenerator:
    def __init__(self, target_lang: str):
        print(f"Initializing QuestionGenerator for language: {target_lang}")
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            print("Warning: OPENROUTER_API_KEY not found in environment variables")
        self.target_lang = target_lang
        self.translate_api_url = 'http://127.0.0.1:5000/translate'
        self.openrouter_api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Ensure data directory exists
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        print("Data directory checked/created")
        
        # Translate "Please answer in English" once at initialization
        self.please_answer = self.translate_text("Please answer in English")
        print(f"Translated prompt: '{self.please_answer}'")

    def translate_text(self, text: str, source_lang: str = "en") -> str:
        """Translate text using the translation API"""
        print(f"Translating text: '{text}' to {self.target_lang}")
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
                result = response.json()['translatedText']
                print(f"Translation successful: '{result}'")
                return result
            print(f"Translation failed with status code: {response.status_code}")
            return text
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def generate_questions_for_noun(self, noun: str) -> List[str]:
        """Generate unique questions about a noun using OpenRouter API"""
        print(f"\nGenerating questions for noun: '{noun}'")
        
        prompt = f"""Generate 10 unique questions in {self.target_lang} about '{noun}'.
        Make each question different and interesting.
        Add "{self.please_answer}" at the end of each question.
        Return just the questions, one per line."""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/lukaszbartoszcze/strong-compute", 
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "mistralai/mistral-7b-instruct",  # or another model
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.9
        }
        
        try:
            response = requests.post(
                self.openrouter_api_url,
                headers=headers,
                json=data
            )
            
            if not response.ok:
                print(f"OpenRouter API error: {response.status_code}")
                print(f"Response: {response.text}")
                return []
                
            questions = response.json()['choices'][0]['message']['content'].split('\n')
            questions = [q.strip() for q in questions if q.strip()]
            print(f"Generated {len(questions)} questions")
            return questions
            
        except Exception as e:
            print(f"OpenRouter API error for noun '{noun}': {e}")
            return []

    def process_nouns_and_generate_questions(self, num_questions_per_noun: int = 10):
        """Main process to generate questions for translated nouns"""
        print(f"\nStarting translation process for nouns...")
        
        # Check if we already have translations
        translations_file = self.data_dir / f"nouns_{self.target_lang}.txt"
        if translations_file.exists():
            print(f"Found existing translations file: {translations_file}")
        else:
            print(f"No existing translations found. Translating nouns...")
            translate_nouns(self.target_lang)
        
        # Read translated nouns
        print("Reading translated nouns...")
        with open(translations_file, 'r', encoding='utf-8') as f:
            translated_nouns = [line.strip() for line in f if line.strip()]
        print(f"Found {len(translated_nouns)} translated nouns")

        # Generate questions for each noun
        all_questions = {}
        for i, noun in enumerate(translated_nouns, 1):
            print(f"\nProcessing noun {i}/{len(translated_nouns)}: {noun}")
            questions = self.generate_questions_for_noun(noun)
            
            # Take only the requested number of questions
            questions = questions[:num_questions_per_noun]
            
            if questions:
                all_questions[noun] = {
                    "translated_noun": noun,
                    "questions": questions
                }
                print(f"Added {len(questions)} questions for {noun}")
            else:
                print(f"No questions generated for {noun}")

            # Add a small delay to avoid rate limits
            time.sleep(0.5)

        # Save results
        output_file = self.data_dir / f"questions_{self.target_lang}.json"
        print(f"\nSaving results to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_questions, f, ensure_ascii=False, indent=2)
        
        print(f"Generated questions saved to {output_file}")
        print(f"Total nouns processed: {len(translated_nouns)}")
        print(f"Total nouns with questions: {len(all_questions)}")
        return all_questions

def generate_questions(target_lang: str, questions_per_noun: int = 10):
    """Convenience function to generate questions"""
    print(f"\nStarting question generation for language: {target_lang}")
    print(f"Questions per noun: {questions_per_noun}")
    generator = QuestionGenerator(target_lang)
    return generator.process_nouns_and_generate_questions(questions_per_noun)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_questions.py <target_language> <questions_per_noun>")
        sys.exit(1)
    
    target_lang = sys.argv[1]
    questions_per_noun = int(sys.argv[2])
    generate_questions(target_lang, questions_per_noun) 