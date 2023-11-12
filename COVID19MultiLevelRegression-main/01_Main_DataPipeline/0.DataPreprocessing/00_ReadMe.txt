FILE LIST:
 - JSONProcessor.py
 	- OBJECTIVE,
    	- This file is for selecting the required infomraiton from the JSON files and so to reduce the size of the datasets;
    - INPUT,
    	- JSON output from Hydrator;
    - OUTPUT,
    	- JSON with selected columns and smaller size;

 - FromJSONtoCSV.ipynb
 	- OBJECTIVE,
 		- This file is for converting the JSON files to the input CSV in the main pipeline;
 		- As the dataset for a single country could be sourced from multiple JSON and the tweets could be a mix with both selected and unselected dates,
 	  	  this .py will also consider the selected date scope for grouping the data and to generate multiple output;
 	- INPUT,
 		- Multiple JSON files;
 		- Selected Date Scope;
 	- OUTPUT,
 		- CSV with the Tweets in selected dates;
 		- CSV with the Tweets out-of-selected dates;

READ ME:
 - Executed JSONProcessor.py before code under FromJSONtoCSV.ipynb;