# ABM - Group 5
## Yara, Max, Mark and Ruben

In this repository, we present the code and project report for the research we conducted into football riots. Based on the research question of which method works best to mitigate football riots, we developed a model based on the original Schelling and Epstein models. We also added a logit-based game to the model.

This README explains the file structure and how to run the simulations.

This project was carried out for the Agent-Based Modelling course at the University of Amsterdam in 2026.

If you want to run all plots script you need to have the data. This is available after emailing: ruben.lanjouw14@gmail.com the zip file is not on git because of it size (530mb zipped).

## How to run it.
### Step 1. Setup Venv

Make a venv. We recommend using Python 3.12.10 or higher.

### Step 2. Download the dependencies

```bash
pip install -r requirements.txt
```
You're all setup! BTW: in the rubric it says we need to list the packages. Please read the requirements.txt if you want to see it. All files are also Black linted on line-length 100 so if you want that as well please install Black and read their documentation.  

## File Structure 

Folders:
- ORIGINAL_MODELS: Legacy code for Schelling and Epstein model. Also contains the code to run the servers.
- RIOT_MODEL: First version of our riot-model. It's not vectorized so a bit slow but with the server you can see some nice simulations!
- SNELLIUS_FILES: Contains the files needed to run big simulations of our model and the script to run it locally. This folder also contains a README with some more explaination how to use it. 
- EXPERIMENTS: Here you can find the code for the plots we made in the report which are not related to the data from snellius. The experiments do have seperate folders to make it easier to run as all models are the same but we adjusted some parameters. 
- DATA_PROCESSING: Contains the scripts used for the Sobol sensitivity analysis and exploratory data analysis presented in the report. We recommend running these scripts from the repository root, with the Snellius output stored in the root-level data folder.

## Run the solara interface

```bash
solara run server_riot_model.py
```

## Responsable AI

We used AI to help us implement our ideas and further develop the model. We tried to use it as responsibly as possible by reviewing the generated code, testing the model, and critically evaluating the assumptions and results.

## Credits and Division of Work

We believe the workload was distributed evenly across the team. Throughout the project, we supported one another where possible and collaborated on all aspects. Although each team member had primary responsibility for specific parts of the project, most components were developed with input from multiple team members.

The main owner roles were divided as follows:

Max: Snellius, Sobol, EDA
Yara: Model, Overal Structure, EDA
Mark: Model, EDA, Vectorization
Ruben: Model, GIT, Snellius

**NOTE:** BECAUSE OF MESSY BRANCHES (DATA PUSHES) WE USED GIT REBASE FOR CLEANING TO NOT GET ALL COMMITS. THIS RESULTS IN HAVING A LOT MORE FILE OWNERSHIP FOR RUBEN WHICH DOES **NOT** REPRESENT THE REAL WORK DONE BY THE INDIVIDUALS.


## Main Literature

1. Epstein, Joshua M. “Modeling Civil Violence: An Agent-Based Computational Approach.” Proceedings of the National Academy of Sciences 99, suppl. 3 (2002): 7243–7250.
2. Schelling, Thomas C. “Dynamic Models of Segregation.” Journal of Mathematical Sociology 1, no. 2 (1971): 143–186.
