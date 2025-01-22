# Data Folder

This folder contains the necessary scripts for preprocessing the Air and Wage datasets.
Additionally, download links to all preprocessed datasets are provided below for direct access.

---

## Datasets Sources

1. **Air Dataset** (Raw Data)  
   Source: [Air Dataset on GitHub](https://github.com/lorismichel/drf/blob/master/applications/air_data/data/datasets/air_data.RData)  
   - Note: This is the raw data before preprocessing.

2. **Wage Dataset** (Raw Data)  
   Source: [Wage Dataset on GitHub](https://github.com/lorismichel/drf/blob/master/applications/wage_data/data/datasets/wage_data.Rdata)  
   - Note: This is the raw data before preprocessing.

3. **DDOP Datasets**
   Original Source for the ddop datasets: https://github.com/d3group/ddop, https://github.com/d3group/A-structured-evaluation-of-data-driven-newsvendor-approaches
---

## Accessing Preprocessed Datasets

This folder contains the preprocessed AIr and Wage datasets as well as the preprocessing. 

To download the preprocessed datasets used in this work use the following `file_id` mappings and generate the corresponding download URL:

```python
# Mapping of dataset names to their Google Drive file IDs
file_id = {
    'bakery': '1r_bDn9Z3Q_XgeTTkJL7352nUG3jkUM0z',
    'yaz': '1xrY3Uv5F9F9ofgSM7dVoSK4bE0gPMg36',
    'm5': '1tCBaxOgE5HHllvLVeRC18zvALBz6B-6w',
    'air': '1DMOaV92n3BFEGeCubaxEys2eLzg2Cic3',
    'wage': '1bn7E7NOoRzE4NwXXs1MYhRSKZHC13qYU',
}[dataset_name]

# Generate a direct download URL
url = f"https://drive.google.com/uc?id={dataset_name}"

--> Replace dataset_name with the desired key from the mapping to get the corresponding dataset.

