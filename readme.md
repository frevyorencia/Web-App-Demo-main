# Web App Demo

A minimal playground that demonstrates how to build and preview simple Streamlit layouts inside an isolated Pipenv environment.

## Prerequisites
- Python 3.10 or newer(3.13.2) installed and available on your PATH
- `pip` available for installing Pipenv

## Set Up the Pipenv Environment
1. Install Pipenv globally if you have not already:
   ```bash
   pip install pipenv
   ```
2. Create the virtual environment and install dependencies (Streamlit is the primary one for this project):
   ```bash
   pipenv install --python 3.13.2 streamlit
   ```
   If a `Pipfile` already exists, simply run `pipenv install` and Pipenv will recreate the environment defined there.
3. Activate the virtual environment whenever you want to work on the project:
   ```bash
   pipenv shell
   ```

## Run the Streamlit Playground
With the environment active (or by prefixing commands with `pipenv run`):
```bash
streamlit run web_practice.py
```
This opens a local web app where you can tweak text snippets, button labels, and basic layout options to experiment with simple website composition.

## Useful Commands
- `pipenv run <command>` – execute a command inside the Pipenv environment without entering the shell.
- `pipenv install <package>` – add another dependency (for example, `pipenv install numpy`).
- `pipenv graph` – inspect the dependency tree.

## Notes
- GitHub: [https://github.com/]
- NVIDIA Model API: [https://build.nvidia.com/nvidia/nv-grounding-dino?snippet_tab=Python]
