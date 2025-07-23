.PHONY: test clean

# to imitate SLURM set only single node
export SLURM_LOCALID=0
# assume you have installed need packages
export SPHINX_MOCK_REQUIREMENTS=0

test: clean
	pip install -q -r requirements.txt
	pip install -q -r _requirements/test.txt

	# use this to run tests
	python -m coverage run --source litai -m pytest src tests -v --flake8
	python -m coverage report

clean:
	# clean all temp runs
	rm -rf $(shell find . -name "mlruns")
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf ./src/*.egg-info
	rm -rf ./build
	rm -rf ./dist
