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
`<pip install -r requirements.txt >`  <br> 
'''
`<pip install -r requirements.txt >`  <br> 
'''
> 一盏灯， 一片昏黄； 一简书， 一杯淡茶。 守着那一份淡定， 品读属于自己的寂寞。 保持淡定， 才能欣赏到最美丽的风景！ 保持淡定， 人生从此不再寂寞。
