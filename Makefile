# Oneshell means I can run multiple lines in a recipe in the same shell, so I don't have to
# chain commands together with semicolon
.ONESHELL:
# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

conda_create:
	conda env create --prefix ./env --file ./environment.yml

conda_update:
	conda env update --prefix ./env --prune --file ./environment.yml

conda_clean:
	conda clean --all

panda_preprocess_database:
	$(CONDA_ACTIVATE) ./env
	python3 -m isupgrader.executors.preprocess_panda_database ./data/raw/panda

panda_tile:
	$(CONDA_ACTIVATE) ./env
	python3 -m isupgrader.executors.tile ./data/raw/panda ./data/processed/panda

panda_processed_clean:
	rm -rf ./data/processed/panda