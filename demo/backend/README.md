# Coordgen Demo Backend

## Setup

Run the following commands:

```sh
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run on local

You can run a local server in debug mode:

```sh
.venv/bin/pip install uvicorn
.venv/bin/uvicorn app:main:app --reload
```
