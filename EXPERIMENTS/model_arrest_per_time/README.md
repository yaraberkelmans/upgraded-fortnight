## Run the tests and get the plots for the distribution of the time to arrest 
Expirement to run the time to arrest stuff.

Run it with:
python run_three_setups.py --workers 8

Locally works fine. Please use 8 workers as we have 10 runs where over we average.

setups.json is used to as parameter settings. So if you want to see different parameters, adjust that file. 

THE PLOTS CAN BE GENERETED WITH THE PLOT FILE
RUN WITH:

python .\EXPERIMENTS\model_arrest_per_time\plot_time_to_arrest.py --data-dir .\EXPERIMENTS\model_arrest_per_time\three_setup_data\ --output-dir PLOTS/distibution_of_arrest
# Disclaimer
Most of the experiment code here is written with instrictions by AI. 