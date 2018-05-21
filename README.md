# Spoken Language Understanding Module for Movie Domain

This repository contains code, evaluation resutls and report of mid-term project for Language Understanding Systems course in the academic year 2017/2018.


### Prerequisites

In order to run this code you need to install OpenGrm and OpenFst tools. Other dependencies are the following python libraries: Pandas, NumPy and scikit-learn.


### Run the code

To recreate all the resultes from the report experiment it's enough to run main.py

```
python main.py
```

All the results will be stored in results/ folder, grouped in different subfolders by model (baseline, minimum/improved, all/cross_vall). Language models of each model will be saved in models/ folder in case user would like to evaluate their perplexity. 

