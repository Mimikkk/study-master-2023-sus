from dataclasses import dataclass
from typing import ClassVar

from pandas import DataFrame, read_csv

@dataclass
class _polls(object):
  year2016: DataFrame
  year2017: DataFrame
  year2018: DataFrame
  year2019: DataFrame
  year2021: DataFrame
  year2022: DataFrame

@dataclass
class _meta(object):
  countries: DataFrame
  contest: DataFrame

def is_english(language: str): return language == 'English'

official_languages = {
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

def is_native(country: str, language: str):
  return official_languages.get(country, []).__contains__(language)


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

      return jury.merge(
        tele.drop(columns=["Total score", "Jury score", "Televoting score"]),
        on=['Contestant'],
        how='inner'
      )

    songs = read('song_data')
    songs.fillna(0, inplace=True)

    songs['in_english'] = songs.language.apply(is_english)
    songs['in_native'] = [
      is_native(country, language)
      for country, language in zip(songs.country, songs.language)
    ]

    return cls(
      _polls(*map(merge_polls, (2016, 2017, 2018, 2019, 2021, 2022))),
      _meta(*map(read, ('country_data', 'contest_data'))),
      songs
    )
