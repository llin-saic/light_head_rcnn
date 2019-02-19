ifeq (${PYTHON},)
PYTHON := python3
endif

all:
    # install pycocotools locally
	${PYTHON} setup.py build_ext --inplace
	rm -rf build

install:
	# install pycocotools to the Python site-packages
	${PYTHON} setup.py build_ext install --user
	rm -rf build
