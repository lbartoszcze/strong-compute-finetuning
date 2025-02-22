import os
import json
import requests
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
import sys
import time
from openai import OpenAI

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
        
        # Initialize OpenAI client with OpenRouter
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )
        
        # Ensure data directory exists
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        print("Data directory checked/created")
        
        # Translate "Please answer in [target_lang]" at initialization
        self.please_answer = self.translate_text(f"Please answer in {self.target_lang}")
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
        """Generate unique questions about a noun using Deepseek via OpenRouter"""
        print(f"\nGenerating questions for noun: '{noun}'")
        
        prompt_template = f"""Generate 10 unique questions about '{noun}' in {self.target_lang} language. Make them unique to {self.target_lang}. 
        Questions must be in {self.target_lang} language only.
        Make each question different and interesting.
        Add "{self.please_answer}" at the end of each question.
        Return just the questions, one per line."""
        
        prompt = self.translate_text(prompt_template)

        try:
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://github.com/lukaszbartoszcze/strong-compute",
                },
                model="deepseek/deepseek-r1:free",
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.9
            )
            
            questions = completion.choices[0].message.content.split('\n')
            questions = [q.strip() for q in questions if q.strip()]
            print(f"\nGenerated {len(questions)} questions:")
            for q in questions:
                print(f"  {q}")
            return questions
            
        except Exception as e:
            print(f"Deepseek API error for noun '{noun}': {e}")
            return []

    def process_nouns_and_generate_questions(self, num_questions_per_noun: int = 10):
        """Main process to generate questions for translated nouns"""
        print(f"\nStarting translation process for nouns...")
        
        # Check if we already have enough translations
        translations_file = self.data_dir / f"nouns_{self.target_lang}.txt"
        if translations_file.exists():
            with open(translations_file, 'r', encoding='utf-8') as f:
                num_nouns = sum(1 for line in f if line.strip())
            if num_nouns >= 5000:
                print(f"Found {num_nouns} existing translations - using those")
            else:
                print(f"Only found {num_nouns} translations - need to translate more")
                translate_nouns(self.target_lang)
        else:
            print(f"No existing translations found. Translating nouns...")
            translate_nouns(self.target_lang)
        
        # Read translated nouns
        print("Reading translated nouns...")
        with open(translations_file, 'r', encoding='utf-8') as f:
            translated_nouns = []
            for line in f:
                noun = line.strip()
                if noun:  # Skip empty lines
                    translated_nouns.append(noun)
        print(f"Found {len(translated_nouns)} translated nouns")

        # Generate questions for each noun and save incrementally
        output_file = self.data_dir / f"questions_{self.target_lang}.json"
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
                
                # Save after each noun is processed
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(all_questions, f, ensure_ascii=False, indent=2)
                print(f"Saved progress to {output_file}")
            else:
                print(f"No questions generated for {noun}")

            time.sleep(0.5)

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