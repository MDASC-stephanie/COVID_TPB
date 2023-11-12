FILE LIST:
 - LabelPolicyAndPlainSA.py
 	- OBJECTIVE,
		- For conducting the policy classification and also the plain sentiment analysis;
		- The .py will handle the data for one country per each execution;
	- INPUT,
		- Tweets for different countries (Input per country);
	- OUTPUT,
		- Tweets for different countries with predicted policy labels and overall sentiment scores (Ouput per country);

 - ABSAAndTPB.py
 	- OBJECTIVE,
		- For conducting the aspect-based sentiment analysis;
		- The .py will handle the data for one country per each execution;
	- INPUT,
		- Tweets for different countries with predicted policy labels and overall sentiment scores (Input per country);
	- OUTPUT,
		- Tweets for different countries with TPB scores (Ouput per country);

 - ToRegressionINput.py
 	- OBJECTIVE,
		- For merging the tweets for different countries with TPB scores to the regression model input;
		- The .py will handle the data for all of the countries per each execution;
	- INPUT,
		- Tweets for different countries with TPB scores (Input in separated but at once);
		- Selected date scope;
		- Policy implementation information;
	- OUTPUT,
		- Regression model input (Output as a whole);

READ ME:
 - Executed LabelPolicyAndPlainSA.py, followed by ABSAAndTPB.py and then ToRegressionINput.py;