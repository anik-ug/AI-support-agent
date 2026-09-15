setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

inspect:
	python scripts/inspect_dataset.py

prepare:
	python scripts/prepare_data.py

index:
	python scripts/build_index.py

run:
	python scripts/run_demo.py "I cannot log in to my account"

all:
	python scripts/run_all.py

test:
	PYTHONPATH=. pytest -q
