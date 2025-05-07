# Reproduction of Text-EGM: [Interpretation of Intracardiac Electrograms Through Textual Representations]
This repository contains our reproduction of the paper "Interpretation of Intracardiac Electrograms Through Textual Representations" by Han et al., 2024. accessible here https://arxiv.org/abs/2402.01115 . The official code base can be found at willxxy/Text-EGM.

## Scope of Reproduction:
Our goal was to reproduce the key components of the model-pipeline, inferred metrics and visualizations against the external dataset used in the paper.
This reproduction does not assess the efficacy or generalizability of this method. 
The reproduced results are completely based on the external dataset alone.

The external dataset used in the paper is from (Goldberger et al., 2000) accessible here: https://physionet.org/content/iafdb/1.0.0/

## Setup Instructions:
You can follow along with the reproduction of this paper in Colab by clicking here.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github.com/spot-p/Text-EGM/blob/main/project_reproduction_driver.ipynb)

The notebook is categorized into appropriate sections as below:

1) Data Aquisition and Preprocessing : Fetches the external dataset, unzips, normalizes and structures and creates train,test,val.
2) Training : Trains one model at a time, for all 4 models. BigBird, LongFormer, Clinical BigBird and Clinical LongFormer.
3) Inference : Runs inference against the trained best checkpoint and appends the results to results.txt
4) Visualization : (Optional) Generates visualizations using integrated gradients which helps with interpretability.
