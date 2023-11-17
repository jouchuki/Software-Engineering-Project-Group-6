from frontend_st import ArxivFE
from arxiv_reco import prerequisites
import sys


def main():
    sys.setrecursionlimit(50000)
    frontend = ArxivFE(batch_size=10)
    keywords = frontend.st_prompt_for_keywords()
    articles, graph = prerequisites(keywords)
    frontend.st_selection_pipeline(articles, graph)


if __name__ == "__main__":
    main()
