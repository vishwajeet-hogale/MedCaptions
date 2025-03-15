# MedCaption: A Deep Learning Framework for Automated Medical Image Captioning

## Authors:
Anandavardhana Hegde, Luca-Alexandru Zamfira, Yogesh Thakku Suresh, Vishwajeet Shivaji Hogale

## Problem Statement:
Develop a deep-learning model for automated medical image captioning to enhance diagnostic accuracy in Oncology, Cardiology, Surgery, and Pathology.

## Objective:
In this project, we aim to utilize and implement a deep-learning model that assists in a better understanding of medical images. This network would assist everyone practicing medicine by helping doctors read medical reports, ensuring no detail is missed, and reducing human error. We propose a deep neural network to learn relevant, human-understandable captions describing the anatomical structures, pathological findings, and clinical details given an input scan.

We cover the following medical specialties for the project:
1. Oncology
2. Cardiology
3. Surgery 
4. Pathology

However, this proposed model has the future scope to understand various other medical specialties provided well-labeled data.

## Importance of Understanding Medical Imaging:
Understanding medical imaging ensures that healthcare professionals can better interpret images, which is crucial for diagnosing diseases, providing effective treatment, and making informed clinical decisions. This ultimately leads to better patient care and treatment outcomes.

## Dataset:
The **MultiCaRe** dataset offers an extensive, multi-modal resource comprising over 75,000 open-access and de-identified case reports. It provides information on clinical cases with images and captions, totaling 130,000 images. It includes diverse medical specialties such as oncology, cardiology, surgery, and pathology. Its design allows seamless mapping between images and their corresponding case details, enabling deep, integrative analyses of clinical scenarios.

In parallel, the **MeDiCaT** dataset aggregates 217K images from 131K open-access biomedical papers and augments these with detailed captions, inline references for a substantial portion of the figures, and manually annotated subfigures and subcaptions.

### References:
- [MultiCaRe Dataset](https://github.com/mauro-nievoff/MultiCaRe_Dataset)
- [PMC Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC10792687/#ack0001)
- [MeDiCaT Dataset](https://paperswithcode.com/paper/medicat-a-dataset-of-medical-images-captions)

## Evaluation Metrics:
For our medical image captioning project, the goal is to generate **clinically accurate, contextually meaningful, and visually relevant** captions. Unlike general image captioning, where fluency and coherence are prioritized, medical captioning demands precision in describing anatomical structures, pathological findings, and clinical details.

We will evaluate our model using the following key metrics:

1. **CIDEr (Consensus-based Image Description Evaluation):**
   - Measures the specificity and relevance of the generated captions using tf-idf weighted n-gram similarity.
   - Ensures that captions are informative and medically relevant rather than generic descriptions.

2. **CheXbert Accuracy (Biomedical Named Entity Recognition - NER):**
   - Measures how accurately the generated captions include important medical terms (e.g., "pulmonary edema," "cardiomegaly") by comparing them to a domain-specific biomedical NER model.
   - Ensures critical diagnostic terms are correctly mentioned.

3. **CLIPScore (or MedCLIP Score for Medical Images):**
   - Quantifies how well the generated caption aligns with the given medical image using vision-language contrastive embeddings.
   - Helps verify that the model does not hallucinate information that isn't present in the scan.

### Why These Metrics?
This combination ensures our model is evaluated on:
- **Linguistic Quality** (CIDEr, BERTScore)
- **Clinical Correctness** (RadGraph F1, CheXbert)
- **Image-Text Alignment** (CLIPScore/MedCLIP)

## Workload Distribution:
Our project on Image Captioning for Medical Images involves multiple key stages, including dataset preprocessing, model development, training, evaluation, and report documentation. 

| Task | Description | Assigned Member | Completion Timeline |
|------|------------|----------------|---------------------|
| **Data Preprocessing & Exploration** | Cleaning, organizing, and performing exploratory data analysis (EDA) on medical images and captions. Implementing augmentation techniques if necessary. | Anand, Luca, Vishwa, Yogesh | Week 1 |
| **Feature Extraction & Data Representation** | Implementing feature extraction using CNNs (e.g., ResNet, VGG) to obtain meaningful image embeddings. Preparing data representations for the captioning model. | Anand, Luca | Week 2 |
| **Model Architecture & Implementation** | Designing and implementing the image captioning model using RNN/Transformer architecture (e.g., LSTM/GRU/Attention-based Transformer). Writing modular and scalable code. | Yogesh, Vishwa | Week 3 |
| **Training & Evaluation** | Setting up training pipelines, defining loss functions, optimizing hyperparameters, training the model, and evaluating using BLEU, METEOR, ROUGE, CIDEr scores. | Yogesh, Vishwa, Luca, Anand | Week 4 |
| **Documentation & Report Writing** | Shared equally among all team members based on contributions. | All Members | End of Project |

## Model Backbone:
- **CNN + GRU units** as the backbone for the baseline accuracy.


