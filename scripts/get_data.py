# Funktion zur dynamischen Erstellung von Dataset-Einstellungen basierend auf den geladenen Daten
def get_dataset_settings(data):
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
        'sid': {
            'file_id': '1J9bPCfeLDH-mbSnvTHRoCva7pl6cXD3_',
            'backscaling_columns': [col for col in data.columns if col.startswith('demand_')] + ['demand'],
            'bool_columns': [],
            'drop_columns': [col for col in data.columns if col.startswith('item_') or col.startswith('store_')] + ['date']
        },
        'air': {
            'file_id': '1SKPpNxulcusNTjRwCC0p3C_XW7aNBNJZ',
            'backscaling_columns': [] ,
            'bool_columns': [],
            'drop_columns': ["counts", "location", "target"]
        },
                'copula': {
            'file_id': '1H5wdJgmxdhbzeS17w0NkRlHRCESEAd-e',
            'backscaling_columns': [] ,
            'bool_columns': [],
            'drop_columns': []
        },
                'wage': {
            'file_id': '1bn7E7NOoRzE4NwXXs1MYhRSKZHC13qYU',
            'backscaling_columns': [] ,
            'bool_columns': [],
            'drop_columns': []
        }
    }

from dddex.crossValidation import groupedTimeSeriesSplit

def preprocess_data(data, demand_columns, bool_columns, drop_columns):
    
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



    