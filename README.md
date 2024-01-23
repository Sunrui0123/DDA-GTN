DDA-GTN: large-scale drug repurposing on drug-gene-disease heterogenous association networks using graph transformers
==
In this work, we first present a benchmark dataset that includes three entities: drugs, genes, and diseases that form a three-layer heterogeneous network, and introduce Graph Transformers Networks to learn the low-dimensional embedded representations of drugs-diseases in the heterogeneous network as a way to predict drug-disease associations. We named this method DDA-GTN.
# 1. File description
## 1.1 Data_sr2
* 5_C_D.csv: drug-disease association <br>  CTD IDs -- MeSH IDs
* 6_C_G.csv: drug-gene association <br>  CTD IDs -- Gene Symbol
* 6_G_D.csv: gene-disease association <br>  Gene Symbol -- MeSH IDs -- InferenceScore
* disease_feature0829.csv: disease feature matrix 2447*881 matrices
* drug_feature0829.csv: drug feature matrix 5975*881 matrices
* gene_feature0829.csv: gene feature matrix 12582*881 matrices
* final_weight09061.csv: weighting matrix <br> row index -- column index -- weight
* node_list0829.csv: It contains all the nodes in the heterogeneous network in the order of drug(CTD IDs), gene(Gene Symbol), and disease(MeSH IDs), and the positions corresponding to the nodes are the indexes that end up in the sparse matrix
* NegativeSample0829.csv: Randomly select as many negative samples as positive samples from the drug-disease association matrix <br> drug index -- disease index
## 1.2 Mdata+GTN2
* MdataNW.py: code to run DDA-GTN dataset
* MdataW.py: code to run the weighted DDA-GTN dataset
* model_ori.py：GTN model
* utils.py：The function in the model implementation
* inits.py：The custom function in the model implementation
* gcn.py：GCNConv
* methods.py：Calculate predicted performance metrics
* casestudy_Mdata.py：Predicting drug-disease associations in the entire data
* Siridataset：Save the 5-fold data divided during the running of the code, as well as the true labels, predicted labels, and predicted scores for the five TESTs
# 2. How to run
The program is written in python 3.9 and to run the code we provide, you need to install the requirements.txt through inputting the following command in command line mode: <br> 
```
pip install -r request.txt
```
# 1.Platform and Dependency
## 1.1 Platform
- ubuntu xx.0x
- RTX 3090(GB)


## 1.2 Dependency
| Requirements      | Release                                |
| --------- | ----------------------------------- |
| CUDA     | 描述特点1的内容                     |
| Python     | 描述特点2的内容                     |
| torch     | 1.11.0                     |
| torch_geometric     | 2.1.0.post1                     |
| torch-scatter     | 1.6.0                     |
| torch-sparse     | 0.6.15                     |
| torch-cluster     | 1.6.0                     |
| pandas     | 1.4.4                     |
| scikit-learn     | 1.1.2                     |
| matplotlib     | 3.6.0                     |

# 2. Project Catalog Structure
## 2.1 src
> This folder stores the code files.
- MdataNW.py
  > This file containing the code for running cross validation for DDA-GTN. During the running of the code, log files are generated and stored in the result/log.txt.
- MdataW.py
  > This file containing the code for running cross validation for DDA-GTN with weights. During the running of the code, log files are generated and stored in the result/log.txt.
- casestudy_Mdata.py
  > This file contains code for predicting new drug-disease associations.
- model.py
  > This file contains the code for the DDA-GTN model.
- utils.py
  > This file contains the function in the model implementation.
- inits.py
  > This file contains the custom function in the model implementation.
- gcn.py
  > This file contains the graph convolution function, which is called at the model function.
- methods.py
  > This file contains the function that calculates the average indicator value for the 50% discount cross validation.
- plt_log.py.py
  > This file contains the code to draw the loss plot.
- Num_right/left_fre0829.csv
  > This file contains the number of times the disease/drug appears in the overall drug-disease association matrix.

## 2.2 data
The data is in zenodo, which contains the input data for Mdataset's 5 cross validation on DDA-GTN.
| Folder name      | Descriptions                                |
| --------- | ----------------------------------- |
| 5_C_D.csv     | drug-disease association <br>  CTD IDs -- MeSH IDs                     |
| 6_C_G.csv     | drug-gene association <br>  CTD IDs -- Gene Symbol                     |
| 6_G_D.csv     | gene-disease association <br>  Gene Symbol -- MeSH IDs -- InferenceScore                    |
| disease_feature0829.csv     | disease feature matrix 2447*881 matrices                     |
| drug_feature0829.csv     | drug feature matrix 5975*881 matrices                     |
| gene_feature0829.csv     | gene feature matrix 12582*881 matrices                     |
| final_weight09061.csv     | weighting matrix <br> row index -- column index -- weight                     |
| node_list0829.csv     | It contains all the nodes in the heterogeneous network in the order of drug(CTD IDs), gene(Gene Symbol), and disease(MeSH IDs), and the positions corresponding to the nodes are the indexes that end up in the sparse matrix                    |
| NegativeSample0829.csv     | Randomly select as many negative samples as positive samples from the drug-disease association matrix <br> drug index -- disease index |

## 2.3 result
  > This file contains the code for the DDA-GTN model.
- Siridataset
  > This folder contains the data divisions, predicted values, true values, and predicted probabilities of the test samples in the cross validation generated by cross_validation.py.
- records
  This folder contains the log files from the five cross-validations.
- c
  > T.






















