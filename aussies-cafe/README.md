# Aussies Cafe

React café website with a Streamlit Community Cloud entry point.

## Run locally

Build the React site, install the Python dependency, then start Streamlit:

```powershell
npm install
npm run build
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

The checked-in `dist/` folder lets Streamlit Cloud run the app without installing Node.js.
After changing the React site, run `npm run build` and commit the updated `dist/` output.

## Deploy on Streamlit Community Cloud

Create an app using:

- Repository: `Mohammed-Ikram-2077/aiml-projects`
- Branch: `aussies-cafe`
- Main file path: `aussies-cafe/streamlit_app.py`

Streamlit reads the Python dependency from `aussies-cafe/requirements.txt`.
