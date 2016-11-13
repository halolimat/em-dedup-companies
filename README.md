# em-dedup-companies

Implementation of the core expectation-maximization algorithm proposed in the following paper:
[Bohannon, P., Dalvi, N., Raghavan, M., & Olteanu, M. (2014). Deduplicating a Places Database. In WWW.](http://wwwconference.org/proceedings/www2014/proceedings/p409.pdf)

Unsupervised technique to learn distributions over words that are core to each company name and those that are "background" words. 
The problem of determining if two companies are the same is then transformed into computing P(core(c1) == core(c2)), 
that is the probability that the core set of words in company1 are equal to that of company2.


Some [slides](https://docs.google.com/presentation/d/1C2tC-zA9LrYv-zE5vzJGFnFJnEM2_2Ebj46fzkveWxI/edit#slide=id.p) describing
the algorithm and highlighting some of the results.
