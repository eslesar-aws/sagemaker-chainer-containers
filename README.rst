===========================
SageMaker Container Example
===========================

Use this package is a template for your framework compatible with sagemaker python sdk and container support.

-------------------------
Installation Requirements
-------------------------

This project was written using python 2.7.

- Docker and Docker compose
- Nvidia Docker and Nvidia Docker compose (GPU only)
- ssh keys to GitHub sagemaker-container-support repo installed in the machine

pip install --upgrade .

------------
How to start
------------

install poc container support
-----------------------------

.. code:: bash

    ./install_container_support

build a docker image with the example code
------------------------------------------

.. code:: bash

    ./build_image

run tests against the built image
---------------------------------

.. code:: bash

    pytest test/container

implement training
------------------

use /src/ml_framework/training.py as a reference on how to implement training


implement serving
------------------

use /src/ml_framework/training.py as a reference on how to implement serving

-----------------
Project structure
-----------------

├── build_image                                 - builds an image using the framework located in /src in a Dockefile
|                                                 located in /docker
|
├── install_container_support                   - install the POC version of container support required by this package
|
|
├── docker                                      - has CPU|GPU Py2|Py3 docker images (missing Py3 and GPU for now)
│   └── python_2
│       ├── Dockerfile.cpu
│       └── Dockerfile.gpu
|
├── examples                                    - examples of frameworks implementations using this template
│   |
│   ├── complete_serving_script.py              - complete serving script using this template
│   ├── complete_training_script.py             - complete training script using this template
│   └── single_page_framework.py                - example of a micro framework written in one file
|
├── src
│   └── ml_framework                            - main example
│       ├── serving.py                          - implements the serving methods of the framework
│       ├── start.py                            - starts the framework and register the serving and training engines
│       └── training.py                         - implements the training methods of the framework
└── test
    ├── container                               - container tests build a py 2 img before every execution
    |   |
    │   ├── test_serving.py                     - serving and prediction tests using all the examples
    |   |
    │   └── test_training.py                    - single instance and distributed training tests using the all the examples
    |   |
    |   |
    │   ├── distributed_training                - additional files required by distributed training tests
    │   │   ├── customer_script.py
    │   │   └── data
    │   │       └── training
    │   │           └── training_data.json
    │   ├── serving                             - additional files required by serving tests
    │   │   ├── model.json
    │   │   ├── model_and_predict.py
    │   │   ├── model_and_transform.py
    │   │   └── model_input_output_predict.py
    │   └── single_instance_training           - additional files require by training tests
    │       ├── customer_script.py
    │       └── data
    │           └── training
    │               └── training_data.json
    ├── unit
    │   └── test_train.py
    └── utils
        └── local_mode.py
