import os
import json
import requests
from pathlib import Path
from typing import List, Dict
import time

TRANSLATE_API_URL = 'http://127.0.0.1:5000/translate'

def translate_nouns(target_lang: str):
    """Translate nouns from nounlist.txt to specified target language"""
    nouns_dir = Path("data")
    input_file = nouns_dir / "nounlist.txt"
    output_file = nouns_dir / f"nouns_{target_lang}.txt"
    
    # Read nouns
    with open(input_file, 'r', encoding='utf-8') as f:
        nouns = [line.strip() for line in f if line.strip()]
    
    print(f"Translating {len(nouns)} nouns to {target_lang}")
    
    # Translate each noun
    translations = []
    for i, noun in enumerate(nouns, 1):
        print(f"Translating {i}/{len(nouns)}: {noun}", end='\r', flush=True)
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
                translations.append(response.json()['translatedText'])
            else:
                translations.append(noun)
                
        except Exception as e:
            translations.append(noun)
        
        time.sleep(0.1)
    
    # Save only translated words
    with open(output_file, 'w', encoding='utf-8') as f:
        for translated in translations:
            f.write(f"{translated}\n")
    
    print(f"\nTranslations saved to {output_file}")
    return True

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python translate_nouns.py <target_language>")
        sys.exit(1)
    
    translate_nouns(sys.argv[1]) 