import os
import json
import requests
from pathlib import Path
from typing import List, Dict

TRANSLATE_API_URL = 'http://127.0.0.1:5000/translate'

def translate_nouns(target_lang: str):
    """Translate nouns from nounlist.txt to specified target language"""
    # Get the directory of nounlist.txt
    nouns_dir = Path("data")
    input_file = nouns_dir / "nounlist.txt"
    
    if not input_file.exists():
        print(f"Error: Could not find {input_file}")
        return
    
    # Read nouns
    with open(input_file, 'r', encoding='utf-8') as f:
        nouns = [line.strip() for line in f if line.strip()]
    
    print(f"Translating {len(nouns)} nouns to {target_lang}")
    
    # Translate each noun
    translations = {}
    for noun in nouns:
        try:
            response = requests.post(
                TRANSLATE_API_URL,
                json={
                    "q": noun,
                    "source": "en",
                    "target": target_lang
                },
                headers={"Content-Type": "application/json"}
            )
            
            if response.ok:
                data = response.json()
                translations[noun] = data.get('translatedText', noun)
            else:
                print(f"Translation failed for '{noun}' with status {response.status_code}")
                translations[noun] = noun
                
        except Exception as e:
            print(f"Error translating '{noun}': {str(e)}")
            translations[noun] = noun
    
    # Save translations
    output_file = nouns_dir / f"nouns_{target_lang}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for original, translated in translations.items():
            f.write(f"{original}\t{translated}\n")
    
    print(f"Translations saved to {output_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python translate_nouns.py <target_language>")
        sys.exit(1)
    
    translate_nouns(sys.argv[1]) 