
import pandas as pd
import argparse

def main():
    ap = argparse.ArgumentParser(description="Convert CSV results to Markdown table")
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()
