from dataclasses import dataclass
import functools
from typing import ClassVar

import pandas
from pandas import DataFrame, read_csv

_official_languages = {
  'Albania': ['Albanian'],
  'Latvia': ['Latvian'],
  'Lithuania': ['Lithuanian'],
  'Switzerland': ['French', 'German', 'Italian'],
  'Slovenia': ['Slovene'],
  'Ukraine': ['Ukrainian'],
  'Bulgaria': ['Bulgarian'],
  'Netherlands': ['Dutch'],
  'Moldova': ['Romanian'],
  'Portugal': ['Portuguese'],
  'Croatia': ['Croatian'],
  'Denmark': ['Danish'],
  'Austria': ['German'],
  'Iceland': ['Icelandic'],
  'Greece': ['Greek'],
  'Norway': ['Norwegian'],
  'Armenia': ['Armenian'],
  'Finland': ['Finnish', 'Swedish'],
  'Israel': ['Hebrew'],
  'Serbia': ['Serbian'],
  'Azerbaijan': ['Azerbaijani'],
  'Georgia': ['Georgian'],
  'Malta': ['Maltese', 'English'],
  'San Marino': ['Italian'],
  'Australia': ['English'],
  'Cyprus': ['Greek', 'Turkish'],
  'Ireland': ['English', 'Irish'],
  'North Macedonia': ['Macedonian'],
  'Estonia': ['Estonian'],
  'Romania': ['Romanian'],
  'Poland': ['Polish'],
  'Montenegro': ['Montenegrin'],
  'Belgium': ['Dutch', 'French', 'German'],
  'Sweden': ['Swedish'],
  'Czech Republic': ['Czech'],
  'Italy': ['Italian', 'Italian[f]'],
  'Spain': ['Spanish'],
  'Germany': ['German'],
  'United Kingdom': ['English'],
  'France': ['French'],
  'Russia': ['Russian'],
  'Hungary': ['Hungarian'],
  'Belarus': ['Belarusian'],
  'Bosnia and Herzegovina': ['Bosnian', 'Croatian', 'Serbian'],
  'Turkey': ['Turkish'],
  'Slovakia': ['Slovak'],
  'Andorra': ['Catalan'],
}

def is_english(language: str):
  return language == 'English'

def is_native(country: str, language: str):
  return _official_languages.get(country, []).__contains__(language)

@dataclass
class _polls(object):
  year2016: DataFrame
  year2017: DataFrame
  year2018: DataFrame
  year2019: DataFrame
  year2021: DataFrame
  year2022: DataFrame
  all: DataFrame

@dataclass
class _meta(object):
  countries: DataFrame
  contest: DataFrame


@dataclass
class EurovisionDataset(object):
  votes: _polls
  meta: _meta
  songs: DataFrame
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
      jury.set_index('Contestant', inplace=True)
      tele.set_index('Contestant', inplace=True)

      merged = jury.merge(
        tele.drop(columns=["Total score", "Jury score", "Televoting score"]),
        left_index=True,
        right_index=True,
        how='inner'
      )

      return merged

    songs = read('song_data')
    songs.fillna(0, inplace=True)

    songs['in_english'] = songs.language.apply(is_english)
    songs['in_native'] = list(map(is_native, songs.country, songs.language))

    songs['semi_place'] = songs['semi_place'].replace('-', 0)
    songs['semi_place'] = songs['semi_place'].astype(int)
    songs['final_place'] = songs['final_place'].replace('-', 0)
    songs['final_place'] = songs['final_place'].astype(int)

    years = list(map(merge_polls, (2016, 2017, 2018, 2019, 2021, 2022)))
    summed = functools.reduce(lambda left, right: left.add(right, fill_value=0), years)
    summed['appearance_count'] = summed.index.map(lambda c: sum(1 for year in years if c in year.index))
    summed = summed.div(summed.appearance_count, axis=0)
    summed.drop(columns=['appearance_count'], inplace=True)


    return cls(
      _polls(*years, summed),
      _meta(*map(read, ('country_data', 'contest_data'))),
      songs
    )
