from dataclasses import dataclass

from pandas import DataFrame

resources_directory = 'resources/datasets/'


@dataclass
class _polls(object):
  _2016: DataFrame
  _2017: DataFrame
  _2018: DataFrame
  _2019: DataFrame
  _2020: DataFrame
  _2021: DataFrame
  _2022: DataFrame

@dataclass
class EurovisionDataset(object):
  countries: DataFrame
  contest: DataFrame
  songs: DataFrame
  jury_votes: _polls
  tele_votes: _polls

  @classmethod
  def load(cls):
    return cls()
