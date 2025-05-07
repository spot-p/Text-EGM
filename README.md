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

1) Clones this repository in Colab and installs dependencies.
2) Data Aquisition and Preprocessing : Fetches the external dataset, unzips, normalizes and structures and creates train,test,val.
3) Training : Trains one model at a time, for all 4 models. BigBird, LongFormer, Clinical BigBird and Clinical LongFormer.
4) Inference : Runs inference against the trained best checkpoint and appends the results to results.txt
5) Visualization : (Optional) Generates visualizations using integrated gradients which helps with interpretability.

# Introduction:
This notebook enables a step wise reproduction of the study presented in the paper:

> **Interpretation of Intracardiac Electrograms Through Textual Representations**
> William Jongwon Han, Diana Gomez, Avi Alok, Chaojing Duan, Michael A. Rosenberg, Douglas Weber, Emerson Liu, Ding Zhao
arXiv:2402.01115
Accepted at CHIL 2024


# Notebook Overview:

This notebook is designed to replicate the experiments and analyses conducted in the above paper within Colab, however it can also be executed in a similar captive environment.

It includes:


*  Data Preprocessing: Fetching loading and preparing the Intracardiac Atrial Fibrillation Database for model training.

*  Training: Fine-tuning 4 masked language models on the processed EGM data.

*  Inference: Assessing model performance based on MSE, MAE and AFib classification tasks.

*  Visualization: Visualizing, the reconstructed signal, attribution and attention scores to understand model inference.


By following this notebook, you can explore the innovative methodology proposed in the paper and gain hands-on experience with the techniques discussed.

**Limitation:** While the paper evaluates performance on both internal and external datasets, only the external dataset is publicly available due to the sensitive nature of clinical data. The public dataset contains only a single (positive) class, which may introduce bias or limit the reproducibility and generalization of the results. Users replicating the study should be aware that this class imbalance can significantly affect both training and evaluation metrics especially the classification metrics.

### Clone this repository

```python
!git clone https://github.com/spot-p/Text-EGM.git
```

### Move into the main directory and install dependencies.

```python
%cd Text-EGM
!pip install -r colab_requirements.txt
```

## Data PreProcessing:

### Download the external dataset from physionet.org

```python
%cd preprocess
#!wget https://physionet.org/static/published-projects/iafdb/intracardiac-atrial-fibrillation-database-1.0.0.zip
!gdown 'https://drive.google.com/uc?export=download&id=1XYnKKhNOMuLk5madEJLH-G4YISidu0R-'
!unzip intracardiac-atrial-fibrillation-database-1.0.0

```

### Execute the preprocessing script

```python
!python preprocess_intra.py --path intracardiac-atrial-fibrillation-database-1.0.0
%cd ..
```

## Training

### Train each of the 4 models with processed data

##### With L4 GPU the average execution time per epoch is 37 minutes

```python
!python train.py --device=cuda:0 --batch=4 --patience=5 --model=big --mask=0.75 --use_ce --norm_loss=0.1 --warmup=500 --epochs=20
```

```python
!python train.py --device=cuda:0 --batch=4 --patience=5 --model=clin_bird --mask=0.75 --use_ce --norm_loss=0.1 --warmup=500 --epochs=20
```

```python
!python train.py --device=cuda:0 --batch=4 --patience=5 --model=long --mask=0.75 --use_ce --norm_loss=0.1 --warmup=500 --epochs=20
```

```python
!python train.py --device=cuda:0 --batch=4 --patience=5 --model=clin_long --mask=0.75 --use_ce --norm_loss=0.1 --warmup=500 --epochs=20
```

## Inference
##### ** The inference metrics (MSE,MAE,Accuracy and derived) for the below combinations would be saved into results.txt in the root folder with associated arguments used.

#### Perform Inference without counter-factuals/ inputs perturbations. The --checkpoint arugment would be different if the above training arguments are changed.

```python
!python inference.py --device=cuda:0 --batch=1 --checkpoint='saved_best_0.0001_4_5_0.01_big_True_0.75_1.0_1.0_False_0.1_False_False_False' --model=big --inference --mask=0.75

```

```python
!python inference.py --device=cuda:0 --batch=1 --checkpoint='saved_best_0.0001_4_5_0.01_clin_bird_True_0.75_1.0_1.0_False_0.1_False_False_False' --model=clin_bird --inference --mask=0.75

```

```python
!python inference.py --device=cuda:0 --batch=1 --checkpoint='saved_best_0.0001_4_5_0.01_long_True_0.75_1.0_1.0_False_0.1_False_False_False' --model=long --inference --mask=0.75

```

```python
!python inference.py --device=cuda:0 --batch=1 --checkpoint='saved_best_0.0001_4_5_0.01_clin_long_True_0.75_1.0_1.0_False_0.1_False_False_False' --model=clin_long --inference --mask=0.75

```

#### Perform inference with combinations of counterfactuals / input perturbations.

```python
# All four with label flipping
!python inference.py --device=cuda:0 --batch=1 --checkpoint='saved_best_0.0001_4_5_0.01_long_True_0.75_1.0_1.0_False_0.1_False_False_False' --model=long --inference --mask=0.75 --LF
!python inference.py --device=cuda:0 --batch=1 --checkpoint='saved_best_0.0001_4_5_0.01_clin_long_True_0.75_1.0_1.0_False_0.1_False_False_False' --model=clin_long --inference --mask=0.75 --LF
!python inference.py --device=cuda:0 --batch=1 --checkpoint='saved_best_0.0001_4_5_0.01_clin_bird_True_0.75_1.0_1.0_False_0.1_False_False_False' --model=clin_bird --inference --mask=0.75 --LF
!python inference.py --device=cuda:0 --batch=1 --checkpoint='saved_best_0.0001_4_5_0.01_big_True_0.75_1.0_1.0_False_0.1_False_False_False' --model=big --inference --mask=0.75 --LF

# All four with token substitution
!python inference.py --device=cuda:0 --batch=1 --checkpoint='saved_best_0.0001_4_5_0.01_long_True_0.75_1.0_1.0_False_0.1_False_False_False' --model=long --inference --mask=0.75 --TS
!python inference.py --device=cuda:0 --batch=1 --checkpoint='saved_best_0.0001_4_5_0.01_clin_long_True_0.75_1.0_1.0_False_0.1_False_False_False' --model=clin_long --inference --mask=0.75 --TS
!python inference.py --device=cuda:0 --batch=1 --checkpoint='saved_best_0.0001_4_5_0.01_clin_bird_True_0.75_1.0_1.0_False_0.1_False_False_False' --model=clin_bird --inference --mask=0.75 --TS
!python inference.py --device=cuda:0 --batch=1 --checkpoint='saved_best_0.0001_4_5_0.01_big_True_0.75_1.0_1.0_False_0.1_False_False_False' --model=big --inference --mask=0.75 --TS

# All four with label flipping and token substitution
!python inference.py --device=cuda:0 --batch=1 --checkpoint='saved_best_0.0001_4_5_0.01_long_True_0.75_1.0_1.0_False_0.1_False_False_False' --model=long --inference --mask=0.75 --LF --TS
!python inference.py --device=cuda:0 --batch=1 --checkpoint='saved_best_0.0001_4_5_0.01_clin_long_True_0.75_1.0_1.0_False_0.1_False_False_False' --model=clin_long --inference --mask=0.75 --LF --TS
!python inference.py --device=cuda:0 --batch=1 --checkpoint='saved_best_0.0001_4_5_0.01_clin_bird_True_0.75_1.0_1.0_False_0.1_False_False_False' --model=clin_bird --inference --mask=0.75 --LF --TS
!python inference.py --device=cuda:0 --batch=1 --checkpoint='saved_best_0.0001_4_5_0.01_big_True_0.75_1.0_1.0_False_0.1_False_False_False' --model=big --inference --mask=0.75 --LF --TS

```

## Visualization

```python
%cd visualize
```

##### To visualize the masked reconstructed signal for each model

```python
!python stitching.py --checkpoint='saved_best_0.0001_4_5_0.01_long_True_0.75_1.0_1.0_False_0.1_False_False_False' --instance=100 --time=1
!python stitching.py --checkpoint='saved_best_0.0001_4_5_0.01_long_True_0.75_1.0_1.0_False_0.1_False_False_False' --instance=100 --time=2

!python stitching.py --checkpoint='saved_best_0.0001_4_5_0.01_clin_long_True_0.75_1.0_1.0_False_0.1_False_False_False' --instance=100 --time=1
!python stitching.py --checkpoint='saved_best_0.0001_4_5_0.01_clin_long_True_0.75_1.0_1.0_False_0.1_False_False_False' --instance=100 --time=2

!python stitching.py --checkpoint='saved_best_0.0001_4_5_0.01_clin_bird_True_0.75_1.0_1.0_False_0.1_False_False_False' --instance=100 --time=1
!python stitching.py --checkpoint='saved_best_0.0001_4_5_0.01_clin_bird_True_0.75_1.0_1.0_False_0.1_False_False_False' --instance=100 --time=2

!python stitching.py --checkpoint='saved_best_0.0001_4_5_0.01_big_True_0.75_1.0_1.0_False_0.1_False_False_False' --instance=100 --time=1
!python stitching.py --checkpoint='saved_best_0.0001_4_5_0.01_big_True_0.75_1.0_1.0_False_0.1_False_False_False' --instance=100 --time=2
```

##### To visualize the token importance for reconstruction and forecast / attribution scores for each model. Switch out the respective checkpoints for attribution scores mapped across time and frequency domain.

```python
!python int_grad.py --checkpoint=saved_best_0.0001_5_3_0.01_long_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=long --afibmask
!python int_grad.py --checkpoint=saved_best_0.0001_5_3_0.01_long_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=long --pre --afibmask
!python int_grad.py --checkpoint=saved_best_0.0001_5_3_0.01_long_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=long
!python int_grad.py --checkpoint=saved_best_0.0001_5_3_0.01_long_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=long --pre

!python int_grad.py --checkpoint=saved_best_0.0001_4_3_0.01_clin_long_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=clin_long --afibmask
!python int_grad.py --checkpoint=saved_best_0.0001_4_3_0.01_clin_long_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=clin_long --pre --afibmask
!python int_grad.py --checkpoint=saved_best_0.0001_4_3_0.01_clin_long_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=clin_long
!python int_grad.py --checkpoint=saved_best_0.0001_4_3_0.01_clin_long_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=clin_long --pre

!python int_grad.py --checkpoint=saved_best_0.0001_4_5_0.01_clin_bird_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=clin_bird --afibmask
!python int_grad.py --checkpoint=saved_best_0.0001_4_5_0.01_clin_bird_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=clin_bird --pre --afibmask
!python int_grad.py --checkpoint=saved_best_0.0001_4_5_0.01_clin_bird_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=clin_bird
!python int_grad.py --checkpoint=saved_best_0.0001_4_5_0.01_clin_bird_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=clin_bird --pre

!python int_grad.py --checkpoint=saved_best_0.0001_4_5_0.01_big_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=big --afibmask
!python int_grad.py --checkpoint=saved_best_0.0001_4_5_0.01_big_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=big --pre --afibmask
!python int_grad.py --checkpoint=saved_best_0.0001_4_5_0.01_big_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=big
!python int_grad.py --checkpoint=saved_best_0.0001_4_5_0.01_big_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=big --pre

```

##### To generate attention scores mapped across time and frequency domain for respective models with perturbed inputs. Switch out the the respective model checkpoints for their respective attention scores.

```python
!python viz_attentions.py --checkpoint=saved_best_0.0001_4_5_0.01_long_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=long --afibmask
!python viz_attentions.py --checkpoint=saved_best_0.0001_4_5_0.01_long_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=long --pre --afibmask
!python viz_attentions.py --checkpoint=saved_best_0.0001_4_5_0.01_long_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=long
!python viz_attentions.py --checkpoint=saved_best_0.0001_4_5_0.01_long_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=long --pre

!python viz_attentions.py --checkpoint=saved_best_0.0001_4_5_0.01_clin_long_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=clin_long --afibmask
!python viz_attentions.py --checkpoint=saved_best_0.0001_4_5_0.01_clin_long_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=clin_long --pre --afibmask
!python viz_attentions.py --checkpoint=saved_best_0.0001_4_5_0.01_clin_long_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=clin_long
!python viz_attentions.py --checkpoint=saved_best_0.0001_4_5_0.01_clin_long_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=clin_long --pre

!python viz_attentions.py --checkpoint=saved_best_0.0001_4_5_0.01_clin_bird_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=clin_bird --afibmask
!python viz_attentions.py --checkpoint=saved_best_0.0001_4_5_0.01_clin_bird_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=clin_bird --pre --afibmask
!python viz_attentions.py --checkpoint=saved_best_0.0001_4_5_0.01_clin_bird_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=clin_bird
!python viz_attentions.py --checkpoint=saved_best_0.0001_4_5_0.01_clin_bird_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=clin_bird --pre

!python viz_attentions.py --checkpoint=saved_best_0.0001_4_5_0.01_big_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=big --afibmask
!python viz_attentions.py --checkpoint=saved_best_0.0001_4_5_0.01_big_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=big --pre --afibmask
!python viz_attentions.py --checkpoint=saved_best_0.0001_4_5_0.01_big_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=big
!python viz_attentions.py --checkpoint=saved_best_0.0001_4_5_0.01_big_True_0.75_1.0_1.0_False_0.1_False_False_False --device=cuda:0 --model=big --pre

```

