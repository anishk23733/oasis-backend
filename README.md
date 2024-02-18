# Backend for Oasis at TreeHacks 2024

First, follow the steps from https://github.com/anishk23733/oasis-scripts to scrape data,
extract key indicators, and create a vector database to use with RAG.

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with the `TOGETHER_API_KEY` environment variable.

Run the backend:
```gunicorn -w 2 -b 0.0.0.0:5000 app:app```

If using cloudflared tunneling,
```./cloudflared tunnel run api```
