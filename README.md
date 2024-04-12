# DLNLP_assignment_24_SN20121776
## Project Description
This is the assignment for ELEC0141 Deep Learning for Natural Language Processing. This projects uses the Yahoo! Answers Topic Classification Dataset Version 2 from https://www.kaggle.com/datasets/yacharki/yahoo-answers-10-categories-for-nlp-csv to perform 10-class text classification. 

Data preprocessing is carried out using the preprocessor.py in the Common folder. The dataset is sampled down to around 44k data points and normalised such that it only contains lemmatised ASCII characters without stop words, punctuations and special characters. The cleaned data is then splitted into training, validation and test subset, and loaded into the models using the custom_dataloader.py

Folder A contains the Pytorch RNN model that uses a recurrent neural network with long-short-term memory (LSTM) or gated recurrent unit (GRU) layers. Folder B contains a Huggingface transfromer that uses the distilbert/distilbert-base-uncased as base model. 

All models are fine-tuned for the dataset and their evaluation results (plots and numerical data) are logged in their corresponding folders. 
## Project Organisation
```
DLNLP_ASSIGNMENT_24_SN20121776
|_ A
|   |_ Outputs                  -- Contains the numerical results of the Pytorch models.
|   |_ Plots                    -- Contains the plots of the Pytorch models.
|   |_ RNN_model.py             -- The recurrent neural network model.
|_ B
|   |_ Outputs                  -- Contains the numerial results of the Huggingface model.
|   |_ Plots                    -- Contains th plots of the Huggingface model.
|   |_ transformer_model.py     -- The transformer model.
|_ Common
|   |_ custom_dataloader.py     -- The customised dataloader.
|   |_ early_stopper.py         -- The criteria for early stopping.
|   |_ evaluation_plots.py      -- The code for all plots.
|   |_ preprocessor.py          -- The code for preprocessing the dataset.
|   |_ processor.py             -- The code for running the Pytorch models.
|_ Datasets
|   |_ preprocessed             -- Contains the preprocessed data.
|   |_ raw                      -- Contains the raw data.
|_ .gitignore                   -- The git ignore file.
|_ main.py                      -- The main file of the project.
|_ README.md                    -- The readme.
|_ Report_DLNLP_24_SN20121776   -- The project report.
|_ requirements.txt             -- The required Python libraries. 
```
## Prerequisites
This project requires the libraries stated in requirements.txt.

## Running the Project
When executing main.py, the model can be specified by the arguements -m and -t.\
-m accepts "RNN" and "Transformer";\
-t accepts "LSTM", "GRU".
```
python main.py -m <model> -t <type>
```
If an invalid model is chosen, the script is terminated.
As the script runs, the selected model is hosted on GPU or CPU (if cuda is unavailable). The training and evaluation process and their results are reported on the terminal.
After completed execution, the generated plots are saved in the corresponding folders. 