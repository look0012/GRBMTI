# GWDMTI

## Overview

MicroRNA (miRNA) interactions with messenger RNA (mRNA) are essential for various biological processes, and accurately predicting these interactions is crucial for understanding their mechanisms. Traditional experimental methods often face limitations, making it increasingly important to develop robust predictive models for identifying potential miRNA targets. Current methods tend to rely solely on potential miRNA target sites and do not fully utilize the entire mRNA sequence, which can lead to a loss of crucial features.

To address these limitations, we introduce GWDMTI, a novel deep learning model designed to enhance the prediction of miRNA-target mRNA interactions. GWDMTI leverages both node and sequence features of miRNA and mRNA, aiming to improve predictive performance by overcoming the shortcomings of existing methods.

## Methodology

- **Feature Extraction**: We utilize RNA2vec to train on RNA data, obtaining RNA word vector representations.
- **Sequence Feature Mining**: Convolutional Neural Networks (CNN) and Bidirectional Gated Recurrent Units (BiGRU) are employed to extract RNA sequence features.
- **Node Features**: GraRep is used to derive node features.
- **Feature Integration**: A Deep Neural Network (DNN) integrates sequence and node features to provide a comprehensive prediction of miRNA-mRNA interactions.

## Performance

The GWDMTI model has demonstrated robust performance on the MTIS-9214 dataset, achieving:
- **Accuracy**: 85.892%
- **AUC (Area Under Curve)**: 0.9389
- **AUPR (Area Under Precision-Recall Curve)**: 0.9392

The model also shows high cross-dataset consistency, highlighting its notable referential value for advancing the study of miRNA-target mRNA interactions and indicating its utility and relevance in the field.

## Installation

Ensure you have Python 3.9 installed. Install the required dependencies using the following commands:

```bash
pip install gensim==4.3.3
pip install keras==2.14.0
pip install numpy==1.23.5
pip install pandas==2.0.3
pip install scipy==1.10.1
pip install tensorflow==2.14.0
