# Intro

GraphBepi is a novel graph-based method for accurate B-cell epitope prediction, which is able to capture
spatial information using the predicted protein structures through the edge-enhanced deep graph neural network.

We recommend you to use the [web server](http://bio-web1.nscc-gz.cn/app/graphbepi) of GraphBepi
if your input is small.

![(Variational) gcn](Framework.png)

# System requirement

GraphBepi is developed under Linux environment with:

- python 3.9.12
- numpy 1.21.5
- pandas 1.4.2
- fair-esm 2.0.0
- torch 1.12.1
- pytorch-lightning 1.6.4
- (optional) esmfold

# Software requirement

To run the full & accurate version of GraphBepi, you need to make sure the following software is in the [mkdssp](./mkdssp) directory:  
[DSSP](https://github.com/cmbi/dssp) (_dssp ver 2.0.4_ is Already in this repository)

# Build dataset

1. `git clone https://github.com/biomed-AI/GraphBepi.git && cd GraphBepi`
2. `python dataset.py --gpu 0`

It will take about 20 minutes to download the pretrained ESM-2 model and an hour to build our dataset with CUDA.

# Run GraphBepi for training

After building our dataset **_BCE_633_**, train the model with default hyper params:

```
python train.py --dataset BCE_633
```

# Run GraphBepi for prediction

1. Please execute the following command directly if you can provide the PDB file.
2. If you do not have a PDB file, you can use [AlphaFold2](http://bio-web1.nscc-gz.cn/app/alphaFold2_bio) to predict the protein structure.

```
python test.py -i pdb_file -p --gpu 0 -o ./output
```

or

We have also deployed a faster structural prediction model [ESMFold](https://github.com/facebookresearch/esm) in our project, so you can process the sequences directly by following the commands below.

```
python test.py -i fasta_file -f --gpu 0 -o ./output
```

## Optional: train/predict with XGBoost (tabular per-residue)

If you prefer a fast, non-deep-learning baseline, you can export per-residue tabular features and train an XGBoost classifier:

- Export features (creates `./tabular/train.npz` and `./tabular/test.npz`):

```
python -c "from utils import export_tabular; export_tabular('./data/BCE_633','./tabular','train'); export_tabular('./data/BCE_633','./tabular','test')"
```

- Train XGBoost and save model (optionally include GNN embeddings with PCA):

```
# without GNN
python train_xgb.py --root ./data/BCE_633 --out ./tabular --split train --out-model ./model/xgb_model.joblib

# with GNN embeddings and PCA to 10 dims
python train_xgb.py --root ./data/BCE_633 --out ./tabular --split train --out-model ./model/xgb_gnn.joblib --use-gnn --pca-dim 10
```

- Predict on test and output per-protein CSVs:

```
# without GNN
python predict_xgb.py --model ./model/xgb_model.joblib --tabular-dir ./tabular --output ./xgb_output

# with GNN merged features (ensure gnn_test.npz exists, or use export_gnn_emb.py first)
python predict_xgb.py --model ./model/xgb_gnn.joblib --tabular-dir ./tabular --output ./xgb_gnn_output --use-gnn --pca-dim 10
```

Notes: exported features include ESM-2 residue embeddings, DSSP features, degree and neighbor-aggregated edge features. The XGBoost scripts are meant as a lightweight alternative to the BiLSTM+GNN pipeline.

# Web server, citation and contact

The GrpahBepi web server is freely available: [interface](http://bio-web1.nscc-gz.cn/app/graphbepi)

Citation:

```

@article{zengys,
  title={Identifying the B-cell epitopes using AlphaFold2 predicted structures and pretrained language model},
  author={Yuansong Zeng, Zhuoyi Wei, Qianmu Yuan, Sheng Chen, Weijiang Yu, Jianzhao Gao, and Yuedong Yang},
  journal={biorxiv},
  year={2022}
 publisher={Cold Spring Harbor Laboratory}
}

```

Contact:  
Zhuoyi Wei (weizhy8@mail2.sysu.edu.cn)
Yuansong Zeng (zengys@mail.sysu.edu.cn)
