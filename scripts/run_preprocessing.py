import argparse
from srs.data_preprocessing import load_dataset, clean_corpus, save_cleaned

def main():
    parser = argparse.ArgumentParser(description="Clean Amharic text data.")
    parser.add_argument('--input', required=True, help='Path to raw input file')
    parser.add_argument('--output', required=True, help='Path to save cleaned file')
    parser.add_argument('--min_len', type=int, default=10, help='Minimum length of line to keep')
    parser.add_argument('--verbose', action='store_true', help='Print stats')

    args = parser.parse_args()

    df = load_dataset(args.input)
    cleaned = clean_corpus(df['text'].tolist(), min_length=args.min_len, verbose=args.verbose)
    save_cleaned(cleaned, args.output)

    print(f"\n✅ Done. Cleaned file saved at: {args.output}")


if __name__ == '__main__':
    main()
