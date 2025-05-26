# Spectral Ranking with Covariates

This repository contains the code for the project **Spectral Ranking with Covariates**.

URL: https://github.com/Chau999/SpectralRankingWithCovariates


### Algorithms and where to find them

You will find implementations of the following ranking algorithms from the following scripts:

```
src/spektrankers.py
# Our proposed methods
├── SVDRankerNormal (SVDRank)
├── SVDRankerCov (SVDCovRank)
├── SVDRankerKCov (Kernelised SVDCovRank)
├── SerialRank (SerialRank)
├── CSerialRank (C-SerialRank)
├── CCARank (CCRank)
├── KCCARank (Kernelised CCRank)
# Spectral ranking benchmarks
├── RankCentrality (Rank Centrality) 
└── DiffusionRankCentrality (Regularised Rank Centrality)

# Probabilistic ranking benchmarks
src/prefkrr.py
└── PreferentialKRR (Bradley Terry with GP link)
src/baselines.py
├── BradleyTerryRanker (Bradley Terry Model)
└── Pairwise_LogisticRegression (Bradley Terry with logistic regressoin)

```
