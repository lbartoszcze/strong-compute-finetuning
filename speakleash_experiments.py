from speakleash import Speakleash
import os

base_dir = os.path.join(os.path.dirname(__file__))
replicate_to = os.path.join(base_dir, "datasets")

sl = Speakleash(replicate_to)

for d in sl.datasets:
    size_mb = round(d.characters/1024/1024)
    print("Dataset: {0}, size: {1} MB, characters: {2}, documents: {3}".format(d.name, size_mb, d.characters, d.documents))