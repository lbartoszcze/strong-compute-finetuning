import os
import json
import csv
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
import time
from openai import OpenAI
import random

load_dotenv()

class ExampleGenerator:
    def __init__(self, target_lang: str):
        print(f"Initializing ExampleGenerator for language: {target_lang}")
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            print("Warning: OPENROUTER_API_KEY not found in environment variables")
        self.target_lang = target_lang
        
        # Initialize OpenAI client with OpenRouter
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )
        
        # Ensure data directory exists
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)

    def clean_question(self, question: str) -> str:
        """Clean and validate question text"""
        # Skip non-questions (brainstorming text)
        if not question.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.')):
            return ""
            
        # Remove numbering
        question = question.split(".", 1)[1].strip()
        return question

    def strip_answer_prompt(self, question: str) -> str:
        """Remove the 'Please answer in' prompt from the end"""
        if "Proszę odpowiedzieć w" in question:
            return question.rsplit("Proszę odpowiedzieć w", 1)[0].strip()
        return question

    def clean_text(self, text: str) -> str:
        """Clean text to ensure valid JSON and remove English"""
        # List of English text markers
        english_markers = [
            'Okay', 'Let me', 'First', 'I know', 'I should', 'Maybe',
            'Looking at', 'The user', 'Now', 'However', 'Also',
            'For example', 'Then', 'So,', 'Wait,', 'Alright',
            'Moving to', 'Next,', 'Finally,', 'Therefore'
        ]

        # Split text into lines and find first Polish line
        lines = text.split('\n')
        polish_text = ""
        
        # Look for Odpowiedź marker first
        for marker in ["**Odpowiedź:**", "Odpowiedź:"]:
            if marker in text:
                polish_text = text.split(marker)[1].strip()
                break
                
        # If no marker found, find first non-English line
        if not polish_text:
            for line in lines:
                line = line.strip()
                if line and not any(marker in line for marker in english_markers):
                    polish_text = line
                    break

        if not polish_text:
            raise ValueError(f"No Polish text found in response:\n{text}")

        # Clean up the text
        text = polish_text.replace('\u0000', '')
        text = text.replace('\u001f', '')
        text = text.replace('空白', '')
        text = text.replace('\ufffd', '')
        text = text.replace('\r', '')
        text = text.replace('\t', ' ')
        
        # Keep newlines for formatting but clean up multiple spaces
        text = '\n'.join(' '.join(line.split()) for line in text.split('\n'))
        
        return text

    def generate_answer(self, question: str) -> str:
        """Generate an answer for a question using Deepseek"""
        print(f"\nGenerating answer for question: '{question}'")
        
        prompt = f"""Answer this question in {self.target_lang} language:
        Question: {question}
        Provide a clear and concise answer."""

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
                temperature=0.7
            )
            
            answer = completion.choices[0].message.content.strip()
            answer = self.clean_text(answer)  # Clean once and done
            
            print(f"Generated answer: {answer}")
            return answer
            
        except Exception as e:
            print(f"Deepseek API error: {e}")
            return ""

    def process_questions_and_generate_examples(self, num_examples: int = 100):
        """Main process to generate examples from randomly selected questions"""
        questions_file = self.data_dir / f"questions_{self.target_lang}.json"
        if not questions_file.exists():
            raise FileNotFoundError(f"Questions file not found: {questions_file}")

        print("Reading questions file...")
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)

        # Collect all valid questions
        all_questions = []
        for noun, data in questions_data.items():
            for question in data['questions']:
                clean_question = self.clean_question(question)
                if clean_question:  # Skip brainstorming text
                    all_questions.append(clean_question)

        # Randomly select questions
        if len(all_questions) > num_examples:
            selected_questions = random.sample(all_questions, num_examples)
        else:
            selected_questions = all_questions

        print(f"Selected {len(selected_questions)} questions randomly")

        # Prepare output JSON file
        output_file = self.data_dir / f"examples_{self.target_lang}.json"
        print(f"Will save examples to {output_file}")

        # Load existing examples if file exists
        examples = []
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                examples = json.load(f)
            print(f"Loaded {len(examples)} existing examples")
        
        # Process each selected question
        for question in selected_questions:
            answer = self.generate_answer(question)
            
            if answer:
                # Remove prompt before saving
                final_question = self.strip_answer_prompt(question)
                examples.append({
                    "question": final_question,
                    "answer": answer
                })
                
                # Save progress after each example
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(examples, f, ensure_ascii=False, indent=2)
            
            time.sleep(0.5)

        print(f"\nExamples saved to {output_file}")

def generate_examples(target_lang: str):
    """Convenience function to generate examples"""
    print(f"\nStarting example generation for language: {target_lang}")
    generator = ExampleGenerator(target_lang)
    generator.process_questions_and_generate_examples()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python generate_examples.py <target_language>")
        sys.exit(1)
    
    target_lang = sys.argv[1]
    generate_examples(target_lang)
