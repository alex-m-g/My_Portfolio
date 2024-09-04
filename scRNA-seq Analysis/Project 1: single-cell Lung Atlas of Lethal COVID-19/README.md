#  A molecular single-cell lung atlas of lethal COVID-19

## Description

This paper generated a single-cell transcriptome lung atlas of COVID-19 using short-PMI autopsy specimens and control lung samples. The analysis provides a broad census of the cellular landscape, cell programs, and cell circuits of lethal COVID-19.

I will utilize the same dataset used in the paper and perform scRNA-seq analysis to emulate the results the cited paper produced.

## Dataset

Status:	Public on Apr 29, 2021

Title: Columbia University/NYP COVID-19 Lung Atlas

Organism: Homo sapiens

Experiment type: Expression profiling by high throughput sequencing

Summary:	We profiled 116,314 cells using snRNA-seq of 20 frozen lungs obtained from 19 COVID-19 decedents and seven control patients with short postmortem interval (PMI) autopsies. The COVID-19 cohort comprises seven female and 12 male decedents, including 13 patients of Hispanic ethnicity, with an age range from 58 to >89 years, who had acquired SARS-CoV-2 infection and succumbed to the disease. The average time from symptom onset to death was 27.5 days (range, 4–63 days). After rapid autopsy with a median PMI of 4 hours (range 2–9 hours), collected tissues were either flash-frozen or frozen following OCT (optimal cutting temperature) embedment and subjected to snRNA-seq using a droplet-based platform (10x Genomics). All included patients had underlying hypertensive disorder and frequently one or more additional co-morbidities associated with increased risk for severe COVID-19.
  	
Overall design:	Single-nuclei RNA sequencing of 116,314 cells from 20 frozen lungs obtained from 19 COVID-19 decedents and seven control patients.

Citation: Melms JC, Biermann J, Huang H, Wang Y, et al. A molecular single-cell lung atlas of lethal COVID-19. Nature 2021 Jul;595(7865):114-119.

PMID: 33915568

File: GSE171524_RAW.tar

## Code Implementation

I used scanpy and Scvi tools to preprocess and integrate the raw patient samples. 

## Results and Analysis
### Cell Type Analysis
![cell_type identification1](https://github.com/user-attachments/assets/5c9058d3-1a4f-47fb-8e72-2c77b4da98a9)
![cell_type identification2](https://github.com/user-attachments/assets/37abf79b-3996-4c3c-b16b-6ab3ebfe8055)
### Cell Type Frequency in Control vs COVID Paitents
![ctr_covid_cell_type_expression Analysis](https://github.com/user-attachments/assets/6cc5f1e7-aa15-4109-850d-0a172fb7b58e)
### Cell Type Frequency Heatmap in AT1 vs. AT2 (Differential Expression)
![differential_expression_heatmap](https://github.com/user-attachments/assets/ea8b6e44-12f7-429a-ac9c-658fc81bb282)
### Gene Ontology Enrichment
Pending...
### Score Gene Signature
Pending...


While preparing for integration, I ran into a MemmoryError as calling an array containing high dimensional data of multiple samples demanded too much RAM than my computer could handle. After aide with colleagues, my computer specifications simply cannot pass the integration section of the code. Solution: Integrate a smaller set of samples. Instead of 26 patient samples, I reduced it to 3 control and 3 COVID patient samples. 

## Computer Specifications:
-  Processor: Intel Core i3
-  RAM: 8GB
-  GPU: None
-  Storage: 512 GB

## Instructions on How to Run the Code

Pending...

## Accomplishments:
- Pre-processing clustering with one sample
- Integration of multiple samples
- Found marker genes
- Plotted said marker genes
- Labeling cells
- Counted the fraction of cells
- Differential expression between two different groups of cells
- Made differential expression heat maps
- Gene ontology
- Keg enrichment
- Comparisons between genes in different conditions
- Statistical testing
- Scored cells based on their expression of gene signatures... and plotted it

## Skills:
- **Programming Languages:** Python (Pandas, Numpy)
- **Data Processing and Analysis:** ScanPy, scVI, SOLO
- **Machine Learning & Deep Learning:** Variational Autoencoders (VAE), scVI model training, doublet detection using SOLO.
- **Bioinformatics:** Gene filtering, Quality control metrics calculations, Differential Expression Analysis.
- **Data Visualization:** Plotting gene markers, differential expression heatmaps.
