# MatchPyramidRS 
**[Deep neural review text interaction for recommendation systems](https://www.sciencedirect.com/science/article/pii/S1568494620309248?casa_token=z4lgGRkD58QAAAAA:Bi6hgrtblPABSv5BHi9LJ7I_aM2v_rwh-PHTMeNY4xf7_LYuP9zdiIwjGvLXbslA3UId_2vTSg)**

## Introduction

This repository contains the implementation for the model proposed in the research paper mentioned above. The paper introduces a novel deep recommendation system that leverages user-item review text interaction to provide better recommendations. The model's contributions are as follows:

1. **Proposing a novel deep recommendation system:** The model is designed to utilize user-item review text interaction, allowing for more personalized and accurate recommendations.

2. **Comparing the performance with state-of-the-art baselines:** The proposed model's performance is compared against several state-of-the-art baselines on different categories of Amazon review datasets. This comparison demonstrates the effectiveness of the model in delivering improved recommendations.

3. **Estimating the relevance degree as a regression problem:** The model estimates the relevance degree between two texts corresponding to the user and item. It treats this estimation as a regression problem, which helps to capture more nuanced relationships.

4. **Superiority in case of data sparsity:** The research showcases how the proposed model outperforms state-of-the-art baselines, particularly when dealing with sparse data scenarios.

## Usage
```bash
python match_pyramid.py

