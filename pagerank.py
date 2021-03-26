import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000

sys.setrecursionlimit(200000)

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    prob_distribution = {}

    # if the current pages has no link
    if (len(corpus[page])) == 0:
        equal_probablity = (damping_factor / len(corpus)) + ((1 - damping_factor) / len(corpus))
        for pages in corpus:
            prob_distribution[pages] = equal_probablity
        return prob_distribution

    
    # probablity for the links in page, only DAMPING FACTOR
    link_probablity = round((damping_factor / len(corpus[page])), 4)
    
    # probablity for all the pages in corpus, 1 - DAMPING FACTOR
    equal_probablity = round(((1 - damping_factor) / len(corpus)), 4)
    
    for links in corpus[page]:
        prob_distribution[links] = equal_probablity + link_probablity
    for pages in corpus:
        if pages not in prob_distribution:
            prob_distribution[pages] = equal_probablity
    
    return prob_distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # to get the first page randomly
    current_page = random.choice(list(corpus.keys()))

    #initializing a list for samples
    samples = []

    # Loop to get 'n' number of samples
    for i in range(n):
        # probablity distribution for current page
        transitions = transition_model(corpus, current_page, damping_factor)
        # randomly choosing the next page based on the probablity distribution
        next_page = random.choices(list(transitions.keys()), weights = list(transitions.values()))[0]
        samples.append(next_page)
        current_page = next_page

    # convert the samples list into a dict with pages and number of times visited
    samples_dict = {}
    for pages in corpus:
        samples_dict[pages] = 0
    for values in samples:
        samples_dict[values] += 1

    # change the dict's values to probablity, divide values by number of samples.
    for keys, values in samples_dict.items():
        samples_dict[keys] = values / n

    return samples_dict


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # dict to store pages i linked TO page p
    cross_link_corpus = {}
    # stores the page ranks of all the pages
    iterative_probablity = {}

    # initializing the pagerank to 1 / N
    size_of_corpus = len(corpus)
    for pages in corpus:
        iterative_probablity[pages] = 1 / size_of_corpus
        cross_link_corpus[pages] = []

    # creating a dict which stores a page and all the pages linked TO it. (backwards)
    for page in cross_link_corpus:
        for keys in corpus:
            if page in corpus[keys]:
                cross_link_corpus[page].append(keys)

    # applying the recursive mathematic formula specified in the 'Understanding' section.
    while True:
        for pages in iterative_probablity:
            ranged_sum = 0
            # if the current page does not have any links.
            if len(cross_link_corpus[pages]) == 0:
                # interpret as having one link to every page and itself.
                for pages in iterative_probablity:
                    number_of_links = len(corpus[pages])
                    if number_of_links != 0:
                        ranged_sum += iterative_probablity[pages] / number_of_links
                continue
            for links in cross_link_corpus[pages]:
                number_of_links = len(corpus[links])
                ranged_sum += iterative_probablity[links] / number_of_links
            page_rank = ((1 - damping_factor) / len(corpus)) + (damping_factor * ranged_sum)
            old_rank = iterative_probablity[pages]
            iterative_probablity[pages] = page_rank
        # to check if convergence is less than 0.001 (change in rank).
        if abs(old_rank - page_rank) < 0.0001:
            break

    return iterative_probablity

if __name__ == "__main__":
    main()
