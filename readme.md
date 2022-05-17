# Decentralmate: a virtual realestate price estimator
## Downloading the data to generate the model
run `generate_parcel_csv.py`
This will download the data as csv files including road parcels, genesis plazas, Decentraland University, Decentraland Convention Center, District X, Gambling District, get transaction information on all previously sold parcels, and also download MANA-USD exchange rate from yahoo finance.
## Generating the model
run `generate_model.py`
This will generate normalized parcel price predictions and save it to `parcel_model.csv`, which is a map of all predicted normalized prices of all parcels. Note this module needs `parcel_model.csv` and `roads.csv` to run.
## Running the streamlit app
run `streamlit run app.py`