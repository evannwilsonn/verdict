# Data

The [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) (Maas et al., ACL 2011) contains 50,000 IMDB reviews:

- 25,000 for training and 25,000 for testing, each split half positive and half negative.
- A review counts as positive if its writer gave 7 or more stars out of 10, and negative at 4 or fewer. Middling reviews aren't included.
- No film has reviews in both the training and the test split.

The reviews aren't copied into this repo. Download them with:

```sh
./scripts/get-data.sh        # macOS / Linux
./scripts/get-data.ps1       # Windows PowerShell
```

That creates `data/aclImdb/{train,test}/{pos,neg}/`, with one text file per review, named `<id>_<stars>.txt`.

**Citation.** Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. 2011. Learning Word Vectors for Sentiment Analysis. In *Proceedings of ACL 2011*, 142–150.
