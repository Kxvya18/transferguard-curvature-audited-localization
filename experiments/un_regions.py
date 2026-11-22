"""UN geoscheme regions for OpenFlights country names (table: lukes/ISO-3166-Countries-with-
Regional-Codes, all/all.csv, commit 145f1ad, in data/raw/global/un_regions.csv)."""
import csv

from experiments.benchmark_localization import PROJECT_ROOT

TO_ISO = {
    "Bolivia": "Bolivia, Plurinational State of", "British Virgin Islands": "Virgin Islands (British)",
    "Brunei": "Brunei Darussalam", "Burma": "Myanmar", "Cape Verde": "Cabo Verde", "Congo (Brazzaville)": "Congo",
    "Congo (Kinshasa)": "Congo, Democratic Republic of the", "Cote d'Ivoire": "Côte d'Ivoire", "Czech Republic": "Czechia",
    "East Timor": "Timor-Leste", "Falkland Islands": "Falkland Islands (Malvinas)", "Iran": "Iran, Islamic Republic of",
    "Johnston Atoll": "United States Minor Outlying Islands", "Laos": "Lao People's Democratic Republic", "Macau": "Macao",
    "Macedonia": "North Macedonia", "Micronesia": "Micronesia, Federated States of",
    "Midway Islands": "United States Minor Outlying Islands", "Moldova": "Moldova, Republic of",
    "Netherlands": "Netherlands, Kingdom of the", "Netherlands Antilles": "Curaçao",
    "North Korea": "Korea, Democratic People's Republic of", "Palestine": "Palestine, State of", "Reunion": "Réunion",
    "Russia": "Russian Federation", "Saint Helena": "Saint Helena, Ascension and Tristan da Cunha",
    "South Korea": "Korea, Republic of", "Svalbard": "Svalbard and Jan Mayen", "Swaziland": "Eswatini",
    "Syria": "Syrian Arab Republic", "Taiwan": "Taiwan, Province of China", "Tanzania": "Tanzania, United Republic of",
    "Turkey": "Türkiye", "United Kingdom": "United Kingdom of Great Britain and Northern Ireland",
    "United States": "United States of America", "Venezuela": "Venezuela, Bolivarian Republic of", "Vietnam": "Viet Nam",
    "Virgin Islands": "Virgin Islands (U.S.)", "Wake Island": "United States Minor Outlying Islands", "West Bank": "Palestine, State of",
}


def regions(names):
    """{OpenFlights name: set of its region, sub-region and intermediate-region names}."""
    table = {r["name"]: r for r in csv.DictReader(open(PROJECT_ROOT / "data" / "raw" / "global" / "un_regions.csv", encoding="utf-8"))}
    out = {}
    for n in names:
        r = table[TO_ISO.get(n, n)]
        out[n] = {r["region"], r["sub-region"], r["intermediate-region"]} - {""}
    return out
