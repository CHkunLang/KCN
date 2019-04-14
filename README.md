# KCN
Project for knowledge-guided convolutional networks

This project is the source code for the paper "Knowledge-guided Convolutional Networks for Chemical-Disease Relation Extraction", which focus on the Chemical-induced Diseases (CID) Relation Extraction subtask in BioCreative V Track 3 CDR Task.

URL for BioCreative V Track 3 CDR Task: http://biocreative.org/tasks/biocreative-v/track3-cdr/
The original data and official evaluation toolkit could be found here.

=============================Introduction of the data=================================
The orginal data is cleaned by us which is packaged in data_clean fold.

CDR_intra_data_clean: The intra sentence level instances (input sequences)
CTD_intra_data_clean: The knowledge data (knowledge representations) for intra sentence level instances 
CDR_inter_data_clean: The inter sentence level instances (input sequences)
CTD_inter_data_clean: The knowledge data (knowledge representations) for inter sentence level instances 
PubGold.txt: Gold standard results for all the instances in CDR dataset.
PubID.txt: ID set for train and development dataset (to split the instances).

=============================Introduction of the code===================================
Version 1:

Experiment requirement:
python >= 3.5
pytorch >= 0.4

main.py: run the KCN model at intra- and inter-sentence levels
KCN_model.py: The code for KCN model
merge_result.py: Merge the intra- and inter-sentence level results
doc_level_evaluation.py: Evaluation for results. Note that it is not availabel until you download the original data and official evaluation toolkit on the website of BioCreative V Track 3.

=============================How to run=======================================
Run main.py for the results, the evaluation part is included in this page.

More codes for preprocessing is under construction and will be available soon.
