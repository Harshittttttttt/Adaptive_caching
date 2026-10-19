import argparse
from semantic_cache.benchmark import run_benchmark


def main():
    parser = argparse.ArgumentParser(description="Run the offline semantic-cache benchmark.")
    parser.add_argument("--dataset", choices=["synthetic", "nltk_movie_reviews"], default="nltk_movie_reviews")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--queries", type=int, default=60)
    parser.add_argument("--output", default="results/run")
    args = parser.parse_args()
    if args.seeds < 1 or args.queries < 1:
        parser.error("--seeds and --queries must be positive")
    run_benchmark(args.dataset, args.seeds, args.queries, args.output)


if __name__ == "__main__":
    main()
