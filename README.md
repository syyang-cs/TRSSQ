

# Efficient Learning-based Top-k Representative Similar Subtrajectory Query
The rise in trajectory data from location technology has enabled applications like traffic prediction and travel time estimates. A key task is finding trajectories similar to a given one, often using trajectory similarity metrics.  Contrary to the extensive research focused on similarity across entire trajectories, this paper delves into subtrajectory similarity within collections containing a large number of trajectories.   We propose the Top-k Representative Similar Subtrajectory Query (TRSSQ for short), with the objective of identifying the top-k subtrajectories most similar to a query within a trajectory set.   To ensure data diversity, we pioneer the concept of Representative Similarity, where only the subtrajectory with the highest degree of similarity to the query is reported from each trajectory, thus avoiding redundancy in the top-k results. In order to address high computation cost challenge, we propose a learning-based framework, leveraging a deep learning model called Representive Similarity Score Estimation (RSSE) to approximate subtrajectory similarity scores efficiently and reduce the candidate set significantly. Empirical evaluations conducted on real world datasets substantiate the efficacy and operational efficiency of our proposed method.

## Introduction
Representive Similarity Score Estimation (RSSE) is a deep learning framework designed for efficient and accurate estimation of subtrajectory similarities within large trajectory datasets. RSSE addresses the challenge of high computational costs in trajectory similarity tasks and introduces the novel concept of Representative Similarity. This method ensures diverse data representation by selecting only the most similar subtrajectory from each trajectory in a set.

## Installation
```bash
# Example installation command
git git@github.com:syyang-cs/TRSSQ.git
cd TRSSQ
tar -zxvf data.tar.gz
cd Code
sh run.sh
