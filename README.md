# Dead Reckoning
> pedestrian dead reckoning using phone sensors.

Indoor positioning method using Pedestrian Dead Reckoning (PDR) based on phone sensors data: Accelerometer, Magnetometer and Gyroscope.
_For more details, please refer to this [Article][wiki]._

![](result.png)

## Installing the requirements

To install all the packages used in this project:

```sh
pip install -r ./requirements.txt
```

## Uploading data sensors

upload the csv files of your data sensors on ```./data/``` file:

- Accelerometer.csv
- Magnetometer.csv
- Gyroscope.csv
- Location.csv


## Usage

To display sensors data on graphs:

```sh
python ./src/plot.py
```

To display compute steps methods on graphs:
```sh
python ./src/computeACC.py
```

To display tracking:
```sh
python ./src/deadreckoning.py
```

<!-- Markdown link & img dfn's -->
[npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square
[npm-url]: https://npmjs.org/package/datadog-metrics
[npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[wiki]: https://www.researchgate.net/publication/336369807_PEDESTRIAN_DEAD_RECKONING_USING_SMARTPHONES_SENSORS_AN_EFFICIENT_INDOOR_POSITIONING_SYSTEM_IN_COMPLEX_BUILDINGS_OF_SMART_CITIES
