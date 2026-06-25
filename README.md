# ABM - Group 5
## Yara, Max, Mark and Ruben

In this repository, we present the code and project report for the research we conducted into football riots. Based on the research question of which method works best to mitigate football riots, we developed a model based on the original Schelling and Epstein models. We also added a logit-based game to the model.

This README explains the file structure and how to run the simulations.

This project was carried out for the Agent-Based Modelling course at the University of Amsterdam in 2026.


## How to run it.
### Step 1. Setup Venv

Make a venv. We recommend using Python 3.12.10 or higher.

### Step 2. Download the dependencies

```bash
pip install -r requirements.txt
```
You're all setup! BTW: in the rubric it says we need to list the packages. Please read the requirements.txt if you want to see it.

## File Structure 

Folders:
- ORIGIINAL_MODELS: Legacy code for Schelling and Epstein model. Also contains the code to run the servers.
- RIOT_MODEL: First version of our riot-model. It's not vectorized so a bit slow but with the server you can see some nice simulations!
- SNELLIUS_FILES: Contains the files needed to run big simulations of our model and the script to run it locally. This folder also contains a README with some more explaination how to use it. 

## Run the solara interface

```bash
solara run server_riot_model.py
```

## Responsable AI

We used AI to help us implement our ideas and further develop the model. We tried to use it as responsibly as possible by reviewing the generated code, testing the model, and critically evaluating the assumptions and results.

## Literature

1. Epstein, Joshua M. “Modeling Civil Violence: An Agent-Based Computational Approach.” Proceedings of the National Academy of Sciences 99, suppl. 3 (2002): 7243–7250.
2. Schelling, Thomas C. “Dynamic Models of Segregation.” Journal of Mathematical Sociology 1, no. 2 (1971): 143–186.

## Licence

MIT Licence

### CREDITS

Max, Yara, Mark, Ruben