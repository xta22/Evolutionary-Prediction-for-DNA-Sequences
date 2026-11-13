# LTEE WSL Environment

Original environment:
~/envs/EvolutionInference

## Rebuild on Ubuntu/WSL

Create an environment using the Python version recorded in
python_version.txt.

python3 -m venv ~/envs/EvolutionInference
source ~/envs/EvolutionInference/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements_frozen.txt

If platform-specific packages fail, inspect the errors
and use requirements_core.txt as a minimal starting point.

## Activate

source ~/envs/EvolutionInference/bin/activate

## Notes

The repository does not contain the virtual environment
binaries, original LTEE raw datasets, SSH keys, or credentials.
