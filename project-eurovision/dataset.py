from dataclasses import dataclass
from typing import ClassVar

from pandas import DataFrame, read_csv

@dataclass
class _polls(object):
  _2016: DataFrame
  _2017: DataFrame
  _2018: DataFrame
  _2019: DataFrame
  _2021: DataFrame
  _2022: DataFrame

@dataclass
class _meta(object):
  countries: DataFrame
  contest: DataFrame
  songs: DataFrame

@dataclass
class EurovisionDataset(object):
  votes: _polls
  meta: _meta
  _resources_directory: ClassVar[str] = 'resources/datasets/eurovision'


  @classmethod
  def load(cls):
    def read(name: str):
      return read_csv(f'{cls._resources_directory}/{name}.csv', sep=',', encoding_errors='ignore')

    def merge_polls(year: int):
      jury = read(f'Final Results/Jury/{year}_jury_results')
      tele = read(f'Final Results/Televote/{year}_televote_results')

      jury.rename(columns={column: f'Jury-{column}' for column in jury.columns[4:]}, inplace=True)
      tele.rename(columns={column: f'Tele-{column}' for column in tele.columns[4:]}, inplace=True)
      jury.fillna(0, inplace=True)
      tele.fillna(0, inplace=True)

      return jury.merge(tele, on=['Contestant'], how='inner')

    return cls(
      _polls(*map(merge_polls, (2016, 2017, 2018, 2019, 2021, 2022))),
      _meta(*map(read, ('country_data', 'contest_data', 'song_data')))
    )
