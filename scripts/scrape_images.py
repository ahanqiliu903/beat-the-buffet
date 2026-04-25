"""Download candidate images per class into ml/data/raw/<class_name>/.

Uses icrawler (Bing) to fetch a batch per query. Expect 30–50% junk —
manually delete non-sushi / wrong-dish images after each run before training.

Install:  pip install icrawler
Usage:    python scripts/scrape_images.py --class salmon_nigiri --count 200
          python scripts/scrape_images.py --all --count 150
"""
import argparse
import csv
from pathlib import Path

from icrawler.builtin import BingImageCrawler


QUERIES = {
    "salmon_nigiri": ["salmon nigiri sushi", "sake nigiri"],
    "tuna_nigiri": ["tuna nigiri sushi", "maguro nigiri"],
    "shrimp_tempura_roll": ["shrimp tempura roll sushi", "ebi tempura maki"],
    "salmon_sashimi": ["salmon sashimi", "sake sashimi slices"],
}


def scrape_class(class_name: str, count: int, out_root: Path) -> None:
    out_dir = out_root / class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    per_query = max(1, count // len(QUERIES[class_name]))
    for query in QUERIES[class_name]:
        crawler = BingImageCrawler(storage={"root_dir": str(out_dir)})
        crawler.crawl(keyword=query, max_num=per_query, file_idx_offset="auto")


def load_class_list(labels_csv: str) -> list[str]:
    with open(labels_csv, newline="") as f:
        return [row["class_name"] for row in csv.DictReader(f)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class", dest="cls", help="single class_name")
    parser.add_argument("--all", action="store_true", help="scrape every class")
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--labels", default="ml/data/labels.csv")
    parser.add_argument("--out", default="ml/data/raw")
    args = parser.parse_args()

    out_root = Path(args.out)
    targets = load_class_list(args.labels) if args.all else [args.cls]
    if not targets or targets == [None]:
        parser.error("provide --class <name> or --all")
    for cls in targets:
        print(f"=== scraping {cls} ===")
        scrape_class(cls, args.count, out_root)
