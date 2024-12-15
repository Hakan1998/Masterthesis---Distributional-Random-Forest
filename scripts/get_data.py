# Funktion zur dynamischen Erstellung von Dataset-Einstellungen basierend auf den geladenen Daten
def get_dataset_settings_singleID(data):
    return {
        'bakery': {
            'file_id': '1r_bDn9Z3Q_XgeTTkJL7352nUG3jkUM0z',
            'backscaling_columns': [col for col in data.columns if col.startswith('demand_')] + ['demand'],
            'bool_columns': ['is_schoolholiday', 'is_holiday', 'is_holiday_next2days'],
            'drop_columns': [col for col in data.columns if col.startswith('item_') or col.startswith('store_')] + ['date']
        },
        'yaz': {
            'file_id': '1xrY3Uv5F9F9ofgSM7dVoSK4bE0gPMg36',
            'backscaling_columns': [col for col in data.columns if col.startswith('demand_')] + ['demand'],
            'bool_columns': [],
            'drop_columns': [col for col in data.columns if col.startswith('item_')]
        },
        'm5': {
            'file_id': '1tCBaxOgE5HHllvLVeRC18zvALBz6B-6w',
            'backscaling_columns': [col for col in data.columns if col.startswith('demand_')] + ['demand'],
            'bool_columns': [],
            'drop_columns': [col for col in data.columns if col.startswith('item_') or col.startswith('store_') or col.startswith('state_')]
        },
        'air': {
            'file_id': '1SKPpNxulcusNTjRwCC0p3C_XW7aNBNJZ',
            'backscaling_columns': [] ,
            'bool_columns': [],
            'drop_columns': ["counts", "location", "target"]
        },
                'wage': {
            'file_id': '1bn7E7NOoRzE4NwXXs1MYhRSKZHC13qYU',
            'backscaling_columns': [] ,
            'bool_columns': [],
            'drop_columns': []
        }
    }


def preprocess_data_singleID(data, demand_columns, bool_columns, drop_columns):
    
    # 1. Rückskalierung der 'demand_'-Spalten und der Target-Spalte 'demand'
    for col in demand_columns:
        if col in data.columns:
            data[col] = data[col] * data['scalingValue']

    data["dayCount"] = data["dayIndex"]
    
    data.drop(columns=drop_columns, inplace=True, errors='ignore')
    data[bool_columns] = data[bool_columns].astype(int)

    # 4. Pivot für das Zielvariable (demand) für 'y'
    y = data.pivot_table(index=['dayIndex', 'label'], columns='id', values='demand').reset_index().set_index('dayIndex')
    
    #5. Aufteilung in Trainings- und Testdaten basierend auf dem 'label'
    train_data = data[data['label'] == 'train']
    test_data = data[data['label'] == 'test']

    # 6. Aufteilen der Zielvariablen in Trainings- und Testdaten
    y_train = y[y['label'] == 'train'].drop(columns=['label'])
    y_test = y[y['label'] == 'test'].drop(columns=['label'])

    # 7. Gruppierung der Daten nach 'id' für die Trainings- und Testdatensätze
    X_train_features = train_data.groupby('id')
    X_test_features = test_data.groupby('id')


    return y, train_data, test_data, X_train_features, X_test_features, y_train, y_test



# Funktion zur dynamischen Erstellung von Dataset-Einstellungen basierend auf den geladenen Daten
def get_dataset_settings_alldata(data):
    return {
        'subset_bakery': {
            'file_id': '1r_bDn9Z3Q_XgeTTkJL7352nUG3jkUM0z',
            'backscaling_columns': [col for col in data.columns if col.startswith('demand_')] + ['demand'],
            'bool_columns': ['is_schoolholiday', 'is_holiday', 'is_holiday_next2days'],
            'drop_columns': [''date', "store_50", "store_71", "store_24", "store_38", "store_27", "store_29", "store_70", "store_17", "store_26", "store_45", "store_48", "store_31", "store_43", "store_3", "store_20", "store_49"],
            'drop_keywords': ["50.0", "71.0", "24.0", "38.0", "27.0", "29.0", "70.0", "17.0", "26.0", "45.0", "48.0", "31.0", "43.0", "3.0", "20.0", "49.0"]
        },
        'yaz': {
            'file_id': '1xrY3Uv5F9F9ofgSM7dVoSK4bE0gPMg36',
            'backscaling_columns': [col for col in data.columns if col.startswith('demand_')] + ['demand'],
            'bool_columns': [],
            'drop_columns': [],
            'drop_keywords': ["None"]
        },
        'subset_m5': {
            'file_id': '1tCBaxOgE5HHllvLVeRC18zvALBz6B-6w',
            'backscaling_columns': [col for col in data.columns if col.startswith('demand_')] + ['demand'],
            'bool_columns': [],
            'drop_columns': ["item_FOODS_3_080", "item_FOODS_3_226", "item_FOODS_3_668", "item_FOODS_3_714", "store_CA_2", "store_CA_4", "store_TX_1", "store_TX_3", "store_WI_2", "store_WI_3"],
            'drop_keywords': ["080", "226", "668", "714", "CA_2", "CA_4", "TX_1", "TX_3", "WI_2", "WI_3"]
        },
        'subset_air': {
            'file_id': '1DMOaV92n3BFEGeCubaxEys2eLzg2Cic3',
            'backscaling_columns': [],
            'bool_columns': [],
            'drop_columns': ["counts", "location", "target","pollutant_max_NO2" ,"pollutant_max_SO2", "pollutant_max_PM2.5", "pollutant_max_CO", "Location 6", "Location 3", "Location 5"],
            'drop_keywords': ["CO", "SO2", "NO2" ,"PM2.5", "Location 6", "Location 3", "Location 5"]
        },
        'wage': {
            'file_id': '1bn7E7NOoRzE4NwXXs1MYhRSKZHC13qYU',
            'backscaling_columns': [],
            'bool_columns': [],
            'drop_columns': [],
            'drop_keywords': ["None"]
        }
    }

def drop_rows_by_keywords(data, column_name, keywords):
    if column_name not in data.columns:
        raise ValueError(f"Spalte '{column_name}' nicht im DataFrame gefunden.")
    keyword_filter = ~data[column_name].str.contains('|'.join(keywords), case=False, na=False)
    return data[keyword_filter]

def preprocess_data_alldata(data, dataset_name, bool_columns, drop_columns, drop_keywords):
    # 1. Rückskalierung der 'demand_'-Spalten und der Target-Spalte 'demand'

    data = data.reset_index()
    data.drop(columns=drop_columns, inplace=True, errors='ignore')

    data[bool_columns] = data[bool_columns].astype(int)

    data['id_for_CV'] = data['id']
    data['id_for_CV'] = data['id_for_CV'].astype(str)      
    data = drop_rows_by_keywords(data, "id_for_CV", drop_keywords)
                                      ########################
    data["dummyID"] = "dummyID"                                           #########################
    data.drop(columns=['id', 'index'], inplace=True)                               #########################

    y = data[["demand", "label", "id_for_CV"]].set_index('id_for_CV')                      ####################
    y.rename(columns={'demand': 'dummyID'}, inplace=True)                 ##################

    train_data = data[data['label'] == 'train']
    test_data = data[data['label'] == 'test']


    # 6. Aufteilen der Zielvariablen in Trainings- und Testdaten
    y_train = y[y['label'] == 'train'].drop(columns=['label'])
    y_test = y[y['label'] == 'test'].drop(columns=['label'])

    # 7. Gruppierung der Daten nach 'id' für die Trainings- und Testdatensätze
    X_train_features = train_data.groupby('dummyID')                      #####
    X_test_features = test_data.groupby('dummyID')                        #####
    display(train_data)

    return y, train_data, test_data, X_train_features, X_test_features, y_train, y_test, data, dataset_name