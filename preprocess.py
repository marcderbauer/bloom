"""
Preprocess a given script
"""
from Levenshtein import distance
from langdetect import detect

from utils import split_dset, save_file
import re
import argparse

#----------------------------------------------------------------------------
#                               ARGPARSE
#----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Clean a file to remove duplicates and foreign sentences')
parser.add_argument("filename", type=str, default="data/combined.txt", help="Path of the input file.")
parser.add_argument("--filter_lang", metavar="lang", type=str, default="en", help="Filters all lines not deemed to be of the given language.")
parser.add_argument("--min_distance", type=int, default=3, help="Filters out all lines with a Levenshtein distance up to this value")
parser.add_argument("--min_words", type=int, default=3, help="Filter all lines which have less than min_words.")
parser.add_argument("--split", type=float, default=0.8, help="Splitpoint for train/test set.")
parser.add_argument("--overwrite", default=False, action='store_true', help="Overwrite existing files")

args = parser.parse_args()

print(args.filename)
#----------------------------------------------------------------------------
#                               FILTERS
#----------------------------------------------------------------------------

def filter_language(lines, language=args.filter_lang):
    before = len(lines)
    filtered = list(filter(lambda x: detect(x) == language, lines))
    print(f"Filtered list to only include lang: {language}. Removed {before - len(filtered)} items.")
    return filtered

# Remove duplicates
def filter_similar(lines, min_distance=args.min_distance):
    seen_lines = []
    num_removed = 0
    for line in lines:
        seen = False
        for seen_line in seen_lines:
            if distance(line, seen_line) <= min_distance:
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
        if len(line.split()) < min_words:
            continue
        processed.append(line)
    print(f"Processed & filtered list with regex. Removed {before - len(processed)} items.")
    #TODO: Ensure that sentences don't have a space at the end before \n
    
    return processed


#----------------------------------------------------------------------------
#                               MAIN
#----------------------------------------------------------------------------

def main():
    with open(args.filename, "r") as f:
        lines = [line.strip("\n") for line in f.readlines()]

    original_len = len(lines)
    filtered = filter_language(lines)
    filtered = filter_regex(filtered)
    filtered = filter_similar(filtered)

    # TODO: Not quite happy with the logic here. Revise
    base = args.filename.split(".txt")[0]
    if args.split:
        train, dev = split_dset(filtered, args.split)
        if args.overwrite:
            save_file("train.txt", train)
            save_file("test.txt", dev)
        else:
            save_file(base + "_train.txt", train)
            save_file(base +"_test.txt", dev)
    else:
        if args.overwrite:
            save_file(args.filename)
        else:
            save_file(base + "_processed.txt")
    
    print(f"\n{'-' * 100}\nOriginal Length: {original_len}\nNew Length: {len(filtered)}\nFiltered {original_len - len(filtered)} elements.")

if __name__ == "__main__":
    main()

