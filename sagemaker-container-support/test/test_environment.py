import json
import logging
import os
import shutil
import subprocess
import tempfile

import pytest
from mock import patch

from container_support import ContainerEnvironment, TrainingEnvironment, HostingEnvironment

INPUT_DATA_CONFIG = {
    "train": {"ContentType": "trainingContentType"},
    "evaluation": {"ContentType": "evalContentType"},
    "Validation": {}
}

HYPERPARAMETERS = {
    ContainerEnvironment.USER_SCRIPT_NAME_PARAM: 'myscript.py',
    ContainerEnvironment.USER_SCRIPT_ARCHIVE_PARAM: 's3://mybucket/code.tar.gz',
    "sagemaker_s3_uri_training": "blah/blah",
    "sagemaker_s3_uri_validation": "xxx/yyy",
    "sagemaker_job_name": "my_job_name",
    'sagemaker_region': 'an-aws-region'
}


def optml(subdirs=[]):
    tmp = tempfile.mkdtemp()
    for d in subdirs:
        os.makedirs(os.path.join(tmp, d))
    return tmp


@pytest.fixture()
def hosting():
    os.environ[ContainerEnvironment.USER_SCRIPT_NAME_PARAM.upper()] = "myscript.py"
    os.environ[ContainerEnvironment.USER_SCRIPT_ARCHIVE_PARAM.upper()] = "s3://mybucket/code.tar.gz"

    d = optml(["model"])
    yield d
    shutil.rmtree(d)


@pytest.fixture()
def training():
    d = optml(['input/data/training', 'input/config', 'model', 'output/data'])

    with open(os.path.join(d, 'input/data/training/data.csv'), 'w') as f:
        f.write('dummy data file')

    _write_resource_config(d, 'algo-1', ['algo-1'])
    _write_config_file(d, 'inputdataconfig.json', INPUT_DATA_CONFIG)
    _write_config_file(d, 'hyperparameters.json', _serialize_hyperparameters(HYPERPARAMETERS))

    yield d
    shutil.rmtree(d)


def test_available_cpus(hosting):
    with patch('multiprocessing.cpu_count') as patched:
        patched.return_value = 16
        env = ContainerEnvironment(hosting)
        assert env.available_cpus == 16


# note: this test will fail if run on machine with gpus
def test_available_gpus_nvidia_error(hosting):
    env = ContainerEnvironment(hosting)
    assert env.available_gpus == 0


def test_available_gpus_nvidia_unexpected_output(hosting):
    with patch('subprocess.check_output') as patched:
        patched.return_value = b'???\n'
        env = ContainerEnvironment(hosting)
        assert env.available_gpus == 0


def test_available_gpus_nvidia_exception(hosting):
    with patch('subprocess.check_output') as patched:
        patched.side_effect = subprocess.CalledProcessError(returncode=-1, cmd="nvidia-smi", output="error!")
        env = ContainerEnvironment(hosting)
        assert env.available_gpus == 0


def test_available_gpus_nvidia(hosting):
    with patch('subprocess.check_output') as patched:
        patched.return_value = b'GPU 0: Tesla K80 (UUID: GPU-051ba9d0-4db4-0c3b-05af-dc0f06d7956f)\n' + \
                               b'GPU 1: Tesla K80 (UUID: GPU-051ba9d0-4db4-0c3b-05af-dc0f06d7956e)\n'
        env = ContainerEnvironment(hosting)
        assert env.available_gpus == 2


# hosting tests


def test_model_server_workers_unset(hosting):
    with patch.dict('os.environ', {'SAGEMAKER_CONTAINER_LOG_LEVEL': '20', 'SAGEMAKER_REGION': 'us-west-2'}):
        with patch('multiprocessing.cpu_count') as mp:
            mp.return_value = 13
            env = HostingEnvironment(hosting)
            assert env.model_server_workers is 13


def test_model_server_workers(hosting):
    with patch.dict('os.environ', {'SAGEMAKER_MODEL_SERVER_WORKERS': '2',
                                   'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                                   'SAGEMAKER_REGION': 'us-west-2'}):
        env = HostingEnvironment(hosting)
        assert env.model_server_workers == 2


def test_container_log_level_unset(hosting):
    with pytest.raises(KeyError):
        HostingEnvironment(hosting)


def test_container_log_level(hosting):
    with patch.dict('os.environ', {'SAGEMAKER_CONTAINER_LOG_LEVEL': '20', 'SAGEMAKER_REGION': 'us-west-2'}):
        env = HostingEnvironment(hosting)
        assert env.container_log_level == logging.INFO


# training tests
class TestTrainingEnvironment(TrainingEnvironment):
    def __train__(self, user_module, training_parameters):
        pass


def test_get_channel_dir(training):
    with patch('os.path.exists') as patched:
        patched.return_value = True
        env = TestTrainingEnvironment(training)
        assert env._get_channel_dir("training") == os.path.join(training, "input", "data", "training", "blah/blah")
        assert env._get_channel_dir("validation") == os.path.join(training, "input", "data", "validation", "xxx/yyy")


def test_training_environment_get_env_variables(training):
    with patch('os.path.exists') as patched:
        patched.return_value = True
        env = TestTrainingEnvironment(training)
        assert os.environ[ContainerEnvironment.JOB_NAME_ENV] == "my_job_name"
        assert os.environ[ContainerEnvironment.CURRENT_HOST_ENV] == env.current_host


def test_get_channel_dir_after_ease_fix(training):
    with patch('os.path.exists') as patched:
        patched.return_value = False
        env = TestTrainingEnvironment(training)
        assert env._get_channel_dir("training") == os.path.join(training, "input", "data", "training")
        assert env._get_channel_dir("validation") == os.path.join(training, "input", "data", "validation")


def test_get_channel_dir_no_s3_uri_in_hp(training):
    with patch('os.path.exists') as patched:
        patched.return_value = True
        _write_config_file(training, 'hyperparameters.json', _serialize_hyperparameters({
            "sagemaker_s3_uri_training": "blah/blah",
            "sagemaker_region": "us-west-2"
        }))
        env = TestTrainingEnvironment(training)
        assert env._get_channel_dir("training") == os.path.join(training, "input", "data", "training", "blah/blah")
        assert env._get_channel_dir("validation") == os.path.join(training, "input", "data", "validation")


def test_channels(training):
    env = TestTrainingEnvironment(training)
    assert env.channels == INPUT_DATA_CONFIG


def test_current_host_unset(training):
    _write_resource_config(training, '', [])
    env = TestTrainingEnvironment(training)
    assert env.current_host == ""


def test_current_host(training):
    env = TestTrainingEnvironment(training)
    assert env.current_host == 'algo-1'


def test_hosts_unset(training):
    _write_resource_config(training, '', [])
    env = TestTrainingEnvironment(training)
    assert env.hosts == []


def test_hosts(training):
    hosts = ['algo-1', 'algo-2', 'algo-3']
    _write_resource_config(training, 'algo-1', hosts)
    env = TestTrainingEnvironment(training)
    assert env.hosts == hosts


def test_hosts_single(training):
    env = TestTrainingEnvironment(training)
    assert env.hosts == ['algo-1']


def test_user_script_archive_training(training):
    env = TestTrainingEnvironment(training)
    assert env.user_script_archive == "s3://mybucket/code.tar.gz"


def test_user_script_name_training(training):
    env = TestTrainingEnvironment(training)
    assert env.user_script_name == "myscript.py"


@patch('tempfile.gettempdir')
@patch('container_support.download_s3_resource')
@patch('container_support.untar_directory')
def test_download_user_module(untar, download_s3, gettemp, training):
    env = TestTrainingEnvironment(training)
    gettemp.return_value = 'tmp'
    env.user_script_archive = 'test.gz'

    env.download_user_module()

    download_s3.assert_called_with('test.gz', 'tmp/script.tar.gz')
    untar.assert_called_with('tmp/script.tar.gz', os.path.join(training, 'code'))


@patch('importlib.import_module')
def test_import_user_module(import_module, training):
    env = TestTrainingEnvironment(training)
    env.import_user_module()
    import_module.assert_called_with('myscript')


@patch('importlib.import_module')
def test_import_user_module_without_py(import_module, training):
    env = TestTrainingEnvironment(training)
    env.user_script_name = 'nopy'
    env.import_user_module()
    import_module.assert_called_with('nopy')


def test_env_is_distributed(training):
    _write_resource_config(training, 'algo-1', ['algo-1', 'algo-2'])

    env = TestTrainingEnvironment(training)
    assert env.distributed


def test_env_is_not_distributed(training):
    _write_resource_config(training, 'algo-1', ['algo-1'])

    env = TestTrainingEnvironment(training)
    assert not env.distributed


def _write_config_file(training, filename, data):
    path = os.path.join(training, "input/config/%s" % filename)
    with open(path, 'w') as f:
        json.dump(data, f)


def _write_resource_config(path, current_host, hosts):
    _write_config_file(path, 'resourceconfig.json', {'current_host': current_host, 'hosts': hosts})


def _serialize_hyperparameters(hp):
    return {str(k): json.dumps(v) for (k, v) in hp.items()}

def _setup_training_structure():
    tmp = tempfile.mkdtemp()
    for d in ['input/data/training', 'input/config', 'model', 'output/data']:
        os.makedirs(os.path.join(tmp, d))

    with open(os.path.join(tmp, 'input/data/training/data.csv'), 'w') as f:
        f.write('dummy data file')

    _write_resource_config(tmp, 'a', ['a', 'b'])
    _write_config_file(tmp, 'inputdataconfig.json', INPUT_DATA_CONFIG)
    _write_config_file(tmp, 'hyperparameters.json', _serialize_hyperparameters(HYPERPARAMETERS))

    return tmp