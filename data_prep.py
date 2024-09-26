# prepare data from raw LAS files
# las files not uploaded for purposes of the confidentiality of the data
# but the script can be applied to any other LAS data set
# with minimal or no modification
# pickle dataframe is provided

import os
import pandas as pd
import glob
import lasio

class LASDataProcessor:
    def __init__(self, directory):
        self.directory = directory
        self.las_files = self._get_las_files()
        self.las_objs = self._read_las_files()
        self.dataframe_1 = None
        self.dataframe_2 = None
        self.dataframe_3 = None
        self.dataframe_4 = None

    def _get_las_files(self):
        las_files = glob.glob(os.path.join(self.directory, "*.las"))
        print(f"Found {len(las_files)} LAS files")
        return las_files

    def _read_las_files(self):
        return [lasio.read(f) for f in self.las_files]

    def concatenate_data(self):
        list_of_las = [w.df() for w in self.las_objs]
        self.dataframe_1 = pd.concat(list_of_las)
        return self.dataframe_1

    def filter_logs(self, required_properties):
        self.dataframe_2 = self.dataframe_1.loc[:, required_properties]
        return self.dataframe_2

    def convert_units(self):
        conversions = {
            'HDT': lambda x: x / 3.28084,
            'HAC': lambda x: x / 3.28084,
            'HRHOB': lambda x: x * 1000,
            'HDEN': lambda x: x * 1000,
            'DEN': lambda x: x * 1000,
        }
        for col, func in conversions.items():
            if col in self.dataframe_2:
                self.dataframe_2[col] = func(self.dataframe_2[col])
        return self.dataframe_2

    def group_logs(self):
        self.dataframe_2['DT'] = self.dataframe_2[['HDT', 'HAC', 'AC']].mean(axis=1, skipna=True)
        self.dataframe_2['RHOB'] = self.dataframe_2[['HRHOB', 'HDEN', 'DEN']].mean(axis=1, skipna=True)
        self.dataframe_2['GRL'] = self.dataframe_2[['HGR', 'GR']].mean(axis=1, skipna=True)
        self.dataframe_2['RES'] = self.dataframe_2[['RDEP', 'HRD']].mean(axis=1, skipna=True)
        self.dataframe_2['NPHI'] = self.dataframe_2[['HNPHI', 'NEU', 'TNPH']].mean(axis=1, skipna=True)

        columns_to_drop = ['HDT', 'HAC', 'AC', 'HRHOB', 'HDEN', 'DEN', 'HGR', 'GR', 'RDEP', 'HRD', 'HNPHI', 'NEU', 'TNPH']
        self.dataframe_3 = self.dataframe_2.drop(columns=columns_to_drop)
        return self.dataframe_3

    def drop_nan_values(self):
        self.dataframe_4a = self.dataframe_3.dropna()
        
        # Sonic and gamma can not be negative;
        # density and resistivity of the earth can not be less than or equal to zero
        self.dataframe_4 = self.dataframe_4a[
            (self.dataframe_4a['DT'] > 0) &
            (self.dataframe_4a['RHOB'] > 0) &
            (self.dataframe_4a['GRL'] >= 0) &
            (self.dataframe_4a['RES'] > 0)
        ]
        print("Dataframe 4 has", len(self.dataframe_4), "rows")
        return self.dataframe_4

# Usage
if __name__ == "__main__":
    directory = r"D:\Kirabo\rvae\raw_data"
    required_properties = ['HDT', 'HAC', 'AC', 'HRHOB', 'HDEN', 'DEN', 'HGR', 'GR', 'RDEP', 'HRD', 'HNPHI', 'NEU', 'TNPH']

    processor = LASDataProcessor(directory)
    processor.concatenate_data()
    processor.filter_logs(required_properties)
    processor.convert_units()
    processor.group_logs()
    dataframe_4 = processor.drop_nan_values()

    # Save the processed dataframe to file for access to other scripts
    dataframe_4.to_pickle("dataframe_4.pkl")
