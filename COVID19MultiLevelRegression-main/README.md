# Programs for COVID19MultiLevelRegression 

- This "Program" Folder included the items as below,
	- 01: Main data pipeline:
		- for selecting the required information and reducing the size of the tweet json files for further processing;
		- for conducting the policy type lebelling;
		- for conducting the plain sentiment analysis;
		- for conducting the aspect-based sentiment analysis;
		- for conducting the behavioural intention scoring under TPB & Policy theories framework;
		- for conbiming the information as a whole for later regression input;
	- 02: Sample data input;
	- 03: Sample data output with predicted policy type labels and sentiment scores;
	- 04: R code for regressions, including:
		- multiple linear regression models for change in daily cases vs 
			- implementation-only
			- sentiment-only
			- sentiment + implementation
		- multlevel models for change in daily cases vs 
			- varying-intercept
			- varying-slope
			- varying intercept & slope
				- independent intercept & slope
				- correlated intercept & slope
	- 05: Benchmark model (implementation - only model) and CNN models
	        - Benchmark model (implementation - only model) purely analyses the relationship between implementaiton status of the COVID-19 policy and the daily
		  number of new cases
		- CNN models incorproate the different regression models explored in the project
	- 06: Ontology of Behavioural Determinants
