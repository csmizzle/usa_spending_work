"""
Utilities for USA spending data

"""
from pandas.api.types import is_string_dtype
from typing import Optional, Tuple
from tqdm import tqdm
from rapidfuzz import fuzz as fuzzy_
import pandas as pd


class FuzzPipe:
    """Prep and execute FuzzyPipe on USASpend dataframe"""

    def __init__(
            self,
            dataframe: pd.DataFrame
    ) -> None:
        self.dataframe = dataframe

    @staticmethod
    def _get_string_cols(
            dataframe: pd.DataFrame
    ) -> list:
        """get string columns to apply needs string functions"""
        return [
            col for col in dataframe.columns
            if is_string_dtype(dataframe[col].dtype)
        ]

    def _stringify_dedupe(
            self,
            strip_chars: list = None
    ) -> pd.DataFrame:
        """Prep data as all strings and dedupe"""
        dataframe = self.dataframe.copy().astype(str)
        # strip chars for floats to string
        if not strip_chars:
            strip_chars = ['.0']
        print(f'[!] Original size: {len(dataframe)}')
        string_cols = self._get_string_cols(dataframe)
        for char in strip_chars:
            dataframe[string_cols] = dataframe[string_cols].apply(lambda x: x.str.strip(char))
        dataframe = dataframe.drop_duplicates(subset=string_cols)
        print(f'[!] Dedup on string fields size: {len(dataframe)}')
        return dataframe

    @staticmethod
    def _group_by_id(
            dataframe: pd.DataFrame,
            identifer: str,
            group_field: str
    ) -> pd.DataFrame:
        """Groupby and check for multiple values"""
        return pd.DataFrame(
            dataframe.groupby(identifer)[group_field].size()
        )

    @staticmethod
    def _get_thresh_identifier(
            dataframe: pd.DataFrame,
            identifer: str,
            thresh: int = 1
    ) -> pd.DataFrame:
        """Threshold filter on pandas grouped result"""
        return dataframe[dataframe[identifer] > thresh]

    def _group_by_thresh(
            self,
            dataframe: pd.DataFrame,
            identifier: str,
            group_field: str,
            thresh: int = 1
    ) -> pd.DataFrame:
        """group by and filter to get values with multiple records and set single records"""
        grouped = self._group_by_id(
            dataframe=dataframe,
            identifer=identifier,
            group_field=group_field
        )
        return self._get_thresh_identifier(
            grouped,
            group_field,
            thresh
        )

    def run(
            self,
            group_id: str,
            count_field: str,
            thresh: int = 1,
            strip_chars: list = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run preprocessing for FuzzUSA"""
        # string and dedupe
        dedup = self._stringify_dedupe(strip_chars)
        multi_ids = self._group_by_thresh(
            dataframe=dedup,
            identifier=group_id,
            group_field=count_field,
            thresh=thresh
        )
        # save both single and multi
        multi_ids_df = dedup[dedup[group_id].isin(list(multi_ids.index))]
        single_ids_df = dedup[~dedup[group_id].isin(list(multi_ids.index))]
        print(f'[!] Multi IDs: {len(multi_ids_df)}')
        print(f'[!] Single IDs: {len(single_ids_df)}')
        return (
            multi_ids_df,
            single_ids_df
        )


class FuzzyUSA:
    """Some fuzzy logic for USA spending data"""

    def __init__(
            self,
            dataframe: pd.DataFrame
    ) -> None:
        self.dataframe = dataframe
        self.missing_keys = list()
        self.match_report = list()

    def _fill_nulls(self) -> None:
        """Normalize missing values"""
        self.dataframe = self.dataframe.replace('', 'None')

    @staticmethod
    def _reconstruct_record(
            string_record: str,
            labels: list,
            sep: str = ' * '
    ) -> Optional[dict]:
        """Reconstruct records from fuzzy matcing"""

        values = [value for value in string_record.split(sep) if len(value) > 0]
        if len(values) == len(labels):
            return dict(zip(labels, values))
        else:
            print('Values and labels not equal ...')
            return None

    def _reconstruct_row(
            self,
            key: str,
            string_record: str,
            key_label: str,
            labels: list,
            sep: str = ' * '
    ) -> dict:
        """Reconstruct a fuzzy row with key label"""
        key = {key_label: key}
        return {**key, **self._reconstruct_record(string_record, labels, sep)}

    @staticmethod
    def _construct_lookup(
            dataframe: pd.DataFrame,
            lookup_col: str,
            sep: str = ' * '
    ) -> dict:
        """Create lookup for fast matching on lookup id from dataframe"""
        fuzz_strings = dict()
        for idx, row in dataframe.iterrows():
            string = ''
            for col in dataframe.columns:
                string += row[col] + sep
            if row[lookup_col] not in fuzz_strings.keys():
                fuzz_strings[row[lookup_col]] = [string]
            else:
                fuzz_strings[row[lookup_col]].append(string)
        return fuzz_strings

    def fuzz_match(
            self,
            key_label: str,
            fuzz_fields: list,
            sep: str = ' * ',
            match_thresh: float = 0.9
    ) -> pd.DataFrame:
        """Implement fuzzy matching and reconstruction of rows"""
        self._fill_nulls()
        print(f'[!] Length before resolving: {len(self.dataframe)}')
        fuzz_strings = self._construct_lookup(
            dataframe=self.dataframe,
            lookup_col=key_label,
            sep=sep
        )
        matches = list()
        for key in tqdm(list(fuzz_strings.keys())):
            # get first string in list
            to_compare = fuzz_strings[key][0]
            if len(fuzz_strings[key]) > 1:
                total_match = 0
                copy = fuzz_strings[key].copy()
                copy.pop(0)
                # compare string to other in shared fields
                for string in copy:
                    match_score = fuzzy_.ratio(
                        to_compare,
                        string
                    )
                    total_match += match_score
                    # update match report for investigations and analysis
                    self.match_report.append({
                        'compare': to_compare,
                        'compared': string,
                        'match': match_score
                    })
                # if avg match is above a given thresh
                if total_match / len(copy) >= match_thresh:
                    matches.append(
                        self._reconstruct_row(
                            key=key,
                            key_label=key_label,
                            string_record=to_compare,
                            labels=fuzz_fields
                        )
                    )

                else:
                    print(f'[!] No matches for {to_compare}')
                    self.missing_keys.append(key)
            # if values with single count, make it in add them to resolved
            # for now ... should investigate why this happens
            else:
                matches.append(
                    self._reconstruct_row(
                        key=key,
                        key_label=key_label,
                        string_record=to_compare,
                        labels=fuzz_fields
                    )
                )
        print(f'[!] Missing keys: {len(self.missing_keys)}')
        matches = pd.DataFrame(matches)
        print(f'[!] Length after resolving: {len(matches)}')
        return matches
