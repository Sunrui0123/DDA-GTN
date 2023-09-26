DDA-GTN
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
* NegativeSample0829.csv: Randomly select as many negative samples as positive samples from the drug-disease association matrix <br> row index -- column index
## 1.1 Mdata+GTN2
