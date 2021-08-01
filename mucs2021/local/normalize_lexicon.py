import argparse
import sys
import unicodedata

# Copyright 2021 Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0


def _load_lexicon(f):
    words = {}
    for l in f:
        word, pron = l.strip().split(None, 1)
        if word not in words:
            words[word] = set()
        words[word].add(pron)
    return words


def _load_map(f):
    word_map = {}
    for l in f:
        word, new_word = l.strip().split(None, 1)
        word_norm = unicodedata.normalize("NFKD", word)
        if word_norm not in word_map:
            word_map[word_norm] = new_word
        else:
            if new_word != word_map[word_norm]:
                print(word_norm, new_word, word_map[word], file=sys.stderr)
    return word_map
     

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('lexicon')
    parser.add_argument('map')
    args = parser.parse_args()

    with open(args.lexicon, encoding='utf-8') as f:
        lexicon = _load_lexicon(f)

    with open(args.map, encoding='utf-8') as f:
        word_map = _load_map(f)

    new_lexicon = {}
    for w in lexicon:
        if w in word_map:
            new_word = word_map.get(w, w).split()
            if len(new_word) == 1:
                if new_word[0] in lexicon:
                    new_lexicon[new_word[0]] = lexicon[new_word[0]]
                else:
                    new_lexicon[new_word[0]] = set()
                new_lexicon[new_word[0]].update(lexicon[w])
            else: 
                for nw in new_word:
                    if nw in lexicon:
                        new_lexicon[nw] = lexicon[nw]
                    else:
                        print("Unk word: ", nw, file=sys.stderr) 
        else:
            new_lexicon[w] = lexicon[w]
        
    for w, prons in sorted(new_lexicon.items(), key=lambda x: x[0]):
        if len(prons) == 0:
            print(w, file=sys.stderr)
        for p in sorted(prons):
            print('{} {}'.format(w, p)) 


if __name__ == "__main__":
    main()
