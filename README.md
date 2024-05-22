# ECPN-HFGF

>code for EC number Prediction Network using Hierarchical Features and Global Features(ECPN-HFGF)

## Quick start

1. Clone the repository

`git clone https://github.com/Zidingyi/ECPN-HFGF.git`

2. Create and activate a conda environment

```
conda env create -f env.yml
conda activate ecpn
```

3. Run ECPN-HFGF

run code on cpu:

`python main.py -i data/example.fasta -o result/ -d cpu -t 0.5`

run code on gpu:

`python main.py -i data/example.fasta -o result/ -d cuda:0 -t 0.5`
