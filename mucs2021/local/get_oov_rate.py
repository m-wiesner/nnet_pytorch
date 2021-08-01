import argparse
import unicodedata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('text')
    parser.add_argument('vocab')
    args = parser.parse_args()

    vocab = set()
    oov = 0.0
    total = 0.0
    with open(args.vocab, encoding='utf-8') as f:
        for l in f:
            word = l.strip().split(None, 1)[0]
            vocab.add(unicodedata.normalize("NFKD", word))

    with open(args.text, encoding='utf-8') as f:
        for l in f:
            uttid, text = l.strip().split(None, 1)
            for w in text.split():
                w_ = unicodedata.normalize("NFKD", w)
                if w_ not in vocab:
                    oov += 1
                total += 1            
    print("{:0.2f}%".format(100 * oov / total))

if __name__ == "__main__":
    main()
