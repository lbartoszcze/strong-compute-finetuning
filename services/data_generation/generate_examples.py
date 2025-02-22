import os
import json
import csv
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
import time
from openai import OpenAI

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
        """Remove numbering and extra whitespace from question"""
        # Remove patterns like "1. ", "2. ", etc.
        if question.strip()[0].isdigit():
            question = question.split(".", 1)[1]
        return question.strip()

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
            print(f"Generated answer: {answer}")
            return answer
            
        except Exception as e:
            print(f"Deepseek API error: {e}")
            return ""

    def process_questions_and_generate_examples(self):
        """Main process to generate examples from questions"""
        # Read questions file
        questions_file = self.data_dir / f"questions_{self.target_lang}.json"
        if not questions_file.exists():
            raise FileNotFoundError(f"Questions file not found: {questions_file}")

        print("Reading questions file...")
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)

        # Prepare output CSV file
        output_file = self.data_dir / f"examples_{self.target_lang}.csv"
        print(f"Will save examples to {output_file}")

        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['noun', 'question', 'answer'])  # Header

            # Process each noun and its questions
            for noun, data in questions_data.items():
                print(f"\nProcessing questions for noun: {noun}")
                for question in data['questions']:
                    # Clean the question
                    clean_question = self.clean_question(question)
                    
                    # Generate answer
                    answer = self.generate_answer(clean_question)
                    
                    if answer:
                        # Save to CSV
                        writer.writerow([noun, clean_question, answer])
                        csvfile.flush()  # Ensure it's written to disk
                    
                    time.sleep(0.5)  # Rate limiting

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
