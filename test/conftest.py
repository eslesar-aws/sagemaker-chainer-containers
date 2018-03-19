import logging
import os
import platform
from os.path import join

import pytest
import shutil
import tempfile

from test.utils import local_mode

logger = logging.getLogger(__name__)
logging.getLogger('boto').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.INFO)
logging.getLogger('factory.py').setLevel(logging.INFO)
logging.getLogger('auth.py').setLevel(logging.INFO)
logging.getLogger('connectionpool.py').setLevel(logging.INFO)

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.dirname(os.path.realpath(__file__))

FRAMEWORK_NAME = "chainer"

def pytest_addoption(parser):
    parser.addoption('--dont-build', '-D', action="store_false")
    parser.addoption('--dont-build-base-image', '-B', action="store_false")
    parser.addoption('--dont-install-container-support', '-C', action="store_false")


@pytest.fixture
def opt_ml():
    tmp = tempfile.mkdtemp()
    os.mkdir(os.path.join(tmp, 'output'))

    # Docker cannot mount Mac OS /var folder properly see
    # https://forums.docker.com/t/var-folders-isnt-mounted-properly/9600
    opt_ml_dir = '/private{}'.format(tmp) if platform.system() == 'Darwin' else tmp
    yield opt_ml_dir

    shutil.rmtree(tmp, True)


@pytest.fixture(scope='session', autouse=True)
def install_container_support(request):
    install = request.config.getoption('--dont-install-container-support')
    if not install:
        local_mode.install_container_support()


@pytest.fixture(scope='session', autouse=True)
def build_base_image(request):
    build_base_image = request.config.getoption('--dont-build-base-image')
    framework_version = '3.4.0'
    if build_base_image:
        return local_mode.build_base_image(py_version=2, framework_name=FRAMEWORK_NAME,
                                           framework_version=framework_version, cwd=join(dir_path, '..'))

    return local_mode.get_base_image_tag(framework_name=FRAMEWORK_NAME, framework_version=framework_version,
                                         py_version=2)


@pytest.fixture(scope='session', autouse=True)
def python2_cpu_img(request):
    build_image = request.config.getoption('--dont-build')
    framework_version = '3.4.0'
    if build_image:
        return local_mode.build_image(py_version=2, framework_name=FRAMEWORK_NAME, framework_version=framework_version,
                                      cwd=join(dir_path, '..'))

    return local_mode.get_image_tag(framework_name=FRAMEWORK_NAME, framework_version=framework_version,
                                    py_version=2)
