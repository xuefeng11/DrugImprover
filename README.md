# DrugImprover
We present DrugImprover, a drug optimization framework designed to improve various properties of an original drug in a robust and efficient manner. Within this workflow, we introduce the Advantage-alignment Policy Optimization (APO) algorithm to utilize the advantage preference to perform direct policy improvement under the guidance of multiple critics.
​
Preprint: TODO
​
## Dataset
- The datasets used in this study can be found in the `data` folder in this repository.
  - `data/ST_MODEL`: The trained model for 3CLPro SARS-CoV-2 protease.
  - `data/ST_MODEL_rtcb`: The trained model for RTCB Human-Ligase cancer target.
  - `data/ml.3CLPro*.csv*`: The training and validation SMILES string data docked on 3CLPro.
  - `data/data_RCTB.csv`: The training and validation SMILES string data docked on RTCB.
  - `data/3CLPro_7BQY_A_1_F.oeb`: The 3CLPro OpenEye receptor file.
  - `data/rtcb-*.oedu`: The RTCB OpenEye receptor file.
​
​
## Installation
On macOS, Linux:
```
python3 -m venv env
source env/bin/activate
# TODO ...
```
​
## Usage
TODO
​
## Citations
If you use our work in your research, please cite these papers:
​
The DrugImprover paper:
```bibtex
TODO
```
​
The RCTB dataset:
```bibtex
@article{kroupova2021molecular,
  title={Molecular architecture of the human tRNA ligase complex},
  author={Kroupova, Alena and Ackle, Fabian and Asanovi{\'c}, Igor and Weitzer, Stefan and Boneberg, Franziska M and Faini, Marco and Leitner, Alexander and Chui, Alessia and Aebersold, Ruedi and Martinez, Javier and others},
  journal={Elife},
  volume={10},
  pages={e71656},
  year={2021},
  publisher={eLife Sciences Publications Limited}
}
```
​
The 3CLPro dataset:
```bibtex
@article{jin2020structure,
  title={Structure of Mpro from SARS-CoV-2 and discovery of its inhibitors},
  author={Jin, Zhenming and Du, Xiaoyu and Xu, Yechun and Deng, Yongqiang and Liu, Meiqin and Zhao, Yao and Zhang, Bing and Li, Xiaofeng and Zhang, Leike and Peng, Chao and others},
  journal={Nature},
  volume={582},
  number={7811},
  pages={289--293},
  year={2020},
  publisher={Nature Publishing Group UK London}
}
@article{clyde2023ai,
  title={AI-accelerated protein-ligand docking for SARS-CoV-2 is 100-fold faster with no significant change in detection},
  author={Clyde, Austin and Liu, Xuefeng and Brettin, Thomas and Yoo, Hyunseung and Partin, Alexander and Babuji, Yadu and Blaiszik, Ben and Mohd-Yusof, Jamaludin and Merzky, Andre and Turilli, Matteo and others},
  journal={Scientific Reports},
  volume={13},
  number={1},
  pages={2105},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```
​
The training data for both datasets is a subset of the ZINC 15 dataset:
```bibtex
@article{sterling2015zinc,
  title={ZINC 15--ligand discovery for everyone},
  author={Sterling, Teague and Irwin, John J},
  journal={Journal of chemical information and modeling},
  volume={55},
  number={11},
  pages={2324--2337},
  year={2015},
  publisher={ACS Publications}
}
```
​
The docking surrogate model:
```bibtex
@inproceedings{vasan2023scalable,
  title={Scalable Lead Prediction with Transformers using HPC resources},
  author={Vasan, Archit and Brettin, Thomas and Stevens, Rick and Ramanathan, Arvind and Vishwanath, Venkatram},
  booktitle={Proceedings of the SC'23 Workshops of The International Conference on High Performance Computing, Network, Storage, and Analysis},
  pages={123--123},
  year={2023}
}
```
