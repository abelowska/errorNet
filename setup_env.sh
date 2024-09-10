#!/usr/bin/bash
set -e

python3.8 -m venv --copies venv
source venv/bin/activate
pip3 install wheel
pip3 install -r requirements.txt
python -m ipykernel install --user --name=net

pip install pre-commit
pre-commit install
deactivate
