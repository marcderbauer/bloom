"""
Preprocess a given script
"""
from Levenshtein import distance
from langdetect import detect
import os

filename="data/titles.txt"

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


def main(filename):

    with open(filename, "r") as f:
        lines = [line for line in f.readlines()]

    original_len = len(lines)
    filtered = filter_language(lines)
    filtered = filter_similar(filtered)

    output_file = filename.split(".txt")[0] + "_formatted.txt"
    with open(output_file, "w") as f:
        for line in filtered:
            f.write(line)
    
    print(f"\n{'-' * 100}\nOriginal Length: {original_len}\nNew Length: {len(filtered)}\nFiltered {original_len - len(filtered)} elements.")

if __name__ == "__main__":
    main(filename)

