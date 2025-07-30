#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = hsi_classifier
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:

	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y

	@echo ">>> conda env created. Activate with: \n conda activate $(PROJECT_NAME)"

## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Delete all compiled Python files (only for Linux)
.PHONY: clean # works only in linux with find functionality
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format





#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) hsi_classifier/make_dataset.py --data-name IP
	$(PYTHON_INTERPRETER) hsi_classifier/make_dataset.py --data-name SA
	$(PYTHON_INTERPRETER) hsi_classifier/make_dataset.py --data-name PU
	$(PYTHON_INTERPRETER) hsi_classifier/make_dataset.py --data-name BS

## Make Interium dataset
.PHONY: interium_data
interium_data: requirements
	$(PYTHON_INTERPRETER) hsi_classifier/interium_data.py --data-name IP
	$(PYTHON_INTERPRETER) hsi_classifier/interium_data.py --data-name SA
	$(PYTHON_INTERPRETER) hsi_classifier/interium_data.py --data-name PU

## Run a specific Python file on hsi_classifier
.PHONY: run
run:
	$(PYTHON_INTERPRETER) -u hsi_classifier/$(file).py




#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
