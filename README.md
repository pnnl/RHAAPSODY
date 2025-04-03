# RHAAPSODY


Disclaimer

This material was prepared as an account of work sponsored by an agency of the United States Government.  Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.
Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.
PACIFIC NORTHWEST NATIONAL LABORATORY
operated by
BATTELLE
for the
UNITED STATES DEPARTMENT OF ENERGY
under Contract DE-AC05-76RL01830

## Description
RHAAPSODY is an analysis pipeline designed for automated processing of RHEED (Reflection High-Energy Electron Diffraction) imagery. The pipeline includes functionalities for image processing, data analysis, visualization, and a messaging interface. This tool aims to streamline the analysis of RHEED patterns, making it easier for researchers to extract meaningful insights from their data.

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)


## Installation

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Running the example script
To run the example script, run the following command in the python environment where the above packages are installed:
```bash
python auto_rheeder.py
```
Note: Please update the values for "root", "experiment" and "loop" variable in 'auto_rheeder.py'.
