### gargle_water_LAMP_trial

This repository holds the de-identified data and the code required to produce the quoted figures and statistical values for our publication:

Self-sampled gargle water direct RT-LAMP as a screening method for the detection of SARS-CoV-2 infections

By Skaiste Arbaciauskaite, Pouya Babakhani, Natalia Sandetskaya, Dalius Vitkus, Ligita Jancoriene, Dovile Karosiene, Dovile Karciauskaite, Birute Zablockiene, and Dirk Kuhlmeier

The files "figure_1.svg", "figure_2.svg", "figure_3.svg" correspond to the figures of the publication. "analysis_results.txt" is a text file containing the main statistical values derived from the raw data. The file "code_for_analysis_and_graphs.py" is the python script that produces the figures and the statistical values quoted in the paper. The file "raw_data.csv" contains the de-identified raw data of the trial.

Note that the python script "code_for_analysis_and_graphs.py" script requires and/or makes use of the followling software and packages:

python (version 3.9.2.) 
  os
  math
  random
  json
  numpy (ver. 1.20.1)
  pandas (ver. 1.2.2)
  scipy (ver. 1.6.1)
  sklearn (ver. 0.24.2)
  plotly (ver. 4.14.3)

