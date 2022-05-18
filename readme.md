# Decentralmate: a virtual realestate price estimator
Decentralmate is a price estimator for single parcels that are currently on sale in Decentraland. The streamlit web application is [here](https://tinyurl.com/decentralmate).  
Many of these listed parcels are asking for exorbitant prices, but what price would they realistically sell at? To answer this question, I built a price estimator model using a random forest trained on features such as the x,y coordinates of the parcel, its proximity to roads, plazas and certain districts. In order to compare the relative value of two parcels in a given time period, I normalized the parcel sale price by the 5-day rolling average sale price of all parcels. The normalized price distribution shows the higher priced parcels near the center:  
<img src="https://user-images.githubusercontent.com/60244043/168943408-8098e013-ba98-49ec-8eff-b5920fff9b9f.png?raw=true" width="500" />  
The resulting model also captures the characteristic that parcel values are higher near the center and near roads and plazas:  
<img src="https://user-images.githubusercontent.com/60244043/168943444-43ec57b4-2f10-44dc-931e-c72336a12ae5.png?raw=true" width="500" />  
Using the data from the past year 2021 as the testing set and all prior data as the training set, the model yields a testing R^2 value of 0.65 with MAPE ~13%.

## Downloading the data to generate the model
run `generate_parcel_csv.py`
This will download the data as csv files including road parcels, genesis plazas, Decentraland University, Decentraland Convention Center, District X, Gambling District, get transaction information on all previously sold parcels, and also download MANA-USD exchange rate from yahoo finance.
## Generating the model
run `generate_model.py`
This will generate normalized parcel price predictions and save it to `parcel_model.csv`, which is a map of all predicted normalized prices of all parcels. Note this module needs `parcel_model.csv` and `roads.csv` to run.
## Running the streamlit app
run `streamlit run app.py`
