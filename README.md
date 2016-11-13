# em-dedup-companies

Implementation of the core expectation-maximization algorithm proposed in the following paper:
[Bohannon, P., Dalvi, N., Raghavan, M., & Olteanu, M. (2014). Deduplicating a Places Database. In WWW.](http://wwwconference.org/proceedings/www2014/proceedings/p409.pdf)

Unsupervised technique to learn distributions over words that are core to each company name and those that are "background" words. 
The problem of determining if two companies are the same is then transformed into computing P(core(c1) == core(c2)), 
that is the probability that the core set of words in company1 are equal to that of company2.

Some [slides](https://docs.google.com/presentation/d/1C2tC-zA9LrYv-zE5vzJGFnFJnEM2_2Ebj46fzkveWxI/edit#slide=id.p) describing
the algorithm and highlighting some of the results.


## Getting started

1. Setup you virtualenv and install necessary requirements (from requirements.txt)

2. Run the Starbucks example (from the paper). Modify em-dedup.py to point to the sample input file by replacing the line:
 `in_file = 'company_small.csv'` with `in_file = 'starbucks_test.csv'`, then run the script: `> python em-dedup.py`.
  Learned probability distributions will be written to a file called `probs.csv` where you can inspect the results.
  
3. To run on a custom dataset, create a new input file (similar to `starbucks_test.csv` with your data, be sure to remove any existing 
'pickle' files (e.g. `core.pickle`), then run the script. monitor the log likelihood value as it gets print each iteration, if you see
large, cyclical swings in this value then the algorithm was unable to converge, you may want to try with a smaller dataset first and
see if your results are adequate before scaling up.
  
