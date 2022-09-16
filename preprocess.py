"""
Preprocess a given script
"""
from Levenshtein import distance
from langdetect import detect

from utils import split_dset, save_file
import re

filename="data/combined.txt"

def filter_language(lines, language="en"):
    before = len(lines)
    filtered = list(filter(lambda x: detect(x) == language, lines))
    print(f"Filtered list to only include lang: {language}. Removed {before - len(filtered)} items.")
    return filtered

# Remove duplicates
def filter_similar(lines, max_distance=3):
    seen_lines = []
    num_removed = 0
    for line in lines:
        seen = False
        for seen_line in seen_lines:
            if distance(line, seen_line) <= max_distance:
                seen = True
                num_removed += 1
                break
        if not seen:
            seen_lines.append(line)

    print(f"Filtered list for similar elements. Removed {num_removed} items.")
    return seen_lines

def filter_regex(lines, min_words = 3):
    #regex to remove anything in brackets or after a "|"" \(.+\)| \|.+
    processed = []
    before = len(lines)
    for line in lines:
        line = re.sub("\(.+\)| \|.+","", line) # remove anything in brackets or after a "|"
        if len(line.split())< min_words:
            continue
        processed.append(line)
    print(f"Processed & filtered list with regex. Removed {before - len(processed)} items.")
    return processed

def main(filename, split=0.8):

    with open(filename, "r") as f:
        lines = [line.strip("\n") for line in f.readlines()]

    original_len = len(lines)
    filtered = filter_language(lines)
    filtered = filter_regex(filtered)
    filtered = filter_similar(filtered)

    base = filename.split(".txt")[0]
    if split:
        train, dev = split_dset(filtered, split)
        save_file(base + "_train.txt", train)
        save_file(base + "_eval.txt", dev)
    else:
        save_file(base + "_processed.txt")
    
    print(f"\n{'-' * 100}\nOriginal Length: {original_len}\nNew Length: {len(filtered)}\nFiltered {original_len - len(filtered)} elements.")

if __name__ == "__main__":
    main(filename)

