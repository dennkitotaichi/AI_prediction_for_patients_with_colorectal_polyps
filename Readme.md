# endoscopic_ai.py

"endoscopic_ai.py" is a python AI script for prediction of the requirement of treatment on a Patients with colorectal polyps.

# Features

This program predicts the requirement for treatment of colorectal polyps from data on medical expenses recorded in the patient's practice.

Simulated data used in the study.

Simulated computer data used for medical billing.

# Requirement

pandas
itertools
numpy
matplotlib
codecs
lightgbm
sklearn
pycaret

# Usage

The actual patient data is not attached to this source code for the protection of personal information.
Instead, it generates sample data that experts have verified to be valid.

Download an original data for sample data generation.
On execution of this program. A sample data will be generated from './data/simulated_dpc_data.csv'.

# Note

The importance of the features was calculated numerically, and the top 50 features with the highest total score after 10 runs were selected as the features to be analyzed by PyCaret.

To compare appropriate AI models and perform machine learning, we used PyCaret, an open source software.

In this source code, we have made PyCaret work up to the point where it can perform compare_models.


# Author

* Taichi Endoh
* Tokeidai Memorial Hospital, Sapporo, Japan
* E-mail takkunn1155@gmail.com


# License
"endoscopic_ai.py" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
