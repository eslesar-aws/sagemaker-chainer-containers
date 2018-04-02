import os
import time
import subprocess
import socket
import logging
import threading

from chainer_framework import run_training

from container_support.app import TrainingEngine

import container_support as cs

logger = logging.getLogger(__name__)

engine = TrainingEngine()

_PORT = 7777

@engine.train()
def train(user_module, training_environment):
    """ Runs training on a user supplied module.

    Training is invoked by calling a "train" function in the user supplied module.
    """
    use_mpi = training_environment.hyperparameters.get('use_mpi', len(training_environment.hosts) > 1)
    if not use_mpi:
        run_training.train()
    else:
        _change_hostname(training_environment.current_host)
        if _is_master_node(training_environment.current_host, training_environment.hosts):
            _wait_for_worker_nodes_to_start_sshd([host for host in training_environment.hosts
                                                  if host != training_environment.current_host])
            _run_mpi_on_all_nodes(training_environment)
        else:
            _start_ssh_daemon()
            _wait_for_training_to_finish(training_environment)


def _change_hostname(current_host):
    return os.system("change-hostname.sh %s" % current_host)


def _is_master_node(current_host, hosts):
    return current_host == _get_master_host_name(hosts)


def _get_master_host_name(hosts):
    return sorted(hosts)[0]


def _run_mpi_on_all_nodes(training_environment):
    threading.Thread(target=_create_socket, args=(_get_master_host_name(training_environment.hosts),)).start()

    mpi_command = _get_mpi_command(training_environment)
    logger.info("mpi_command: " + mpi_command)
    subprocess.check_call(mpi_command.split())


def _create_socket(hostname):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((hostname, _PORT))
    server_socket.listen(5)
    server_socket.accept()


def _get_mpi_command(training_environment):
    num_gpus = training_environment.available_gpus
    hyperparameters = training_environment.hyperparameters
    process_slots_per_host = hyperparameters.get('process_slots_per_host', num_gpus if num_gpus > 0 else 1)

    num_hosts = len(training_environment.hosts)
    num_processes = hyperparameters.get('num_processes', process_slots_per_host * num_hosts)
    host_list = training_environment.hosts if process_slots_per_host == 1 else \
        [host + ':{}'.format(process_slots_per_host) for host in training_environment.hosts]
    mpi_command = 'mpirun --allow-run-as-root --host {}'.format(",".join(host_list)) \
                  + " -mca btl_tcp_if_include {0}".format(training_environment.network_interface_name) \
                  + " -mca oob_tcp_if_include {0}".format(training_environment.network_interface_name) \
                  + " -mca btl ^openib" \
                  + " -x PATH" \
                  + " -x LD_LIBRARY_PATH" \
                  + " -x LD_PRELOAD=/libchangehostname.so" \
                  + " -mca orte_abort_on_non_zero_status 1" \
                  + " -x NCCL_DEBUG=INFO" \
                  + " -x NCCL_SOCKET_IFNAME={}".format(training_environment.network_interface_name) \
                  + " -np {} ".format(num_processes) \
                  + "python -m chainer_framework.run_training"
    return mpi_command


def _start_ssh_daemon():
    subprocess.check_call(["/usr/sbin/sshd", "-D"])


def _wait_for_training_to_finish(training_environment):

    current_host = training_environment.current_host

    logger.debug("worker node {} is waiting for MPI to start training process ".format(current_host))

    master_host = _get_master_host_name(training_environment.hosts)

    _wait_for_mpi_to_start_running(master_host)
    logger.debug("MPI started training process on worker node {}".format(current_host))

    _wait_until_mpi_stops_running(master_host)
    logger.debug("Training process started by MPI on worker node {} stopped" .format(current_host))


def _wait_for_worker_nodes_to_start_sshd(hosts, interval=1, timeout=180):
    '''
    "master" node needs to wait for other nodes to be sshable before running MPI.
    '''

    time_elapsed = 0
    while len(hosts) > 0 and time_elapsed < timeout:
        logger.info("hosts that aren't SSHable yet: " + str(hosts))
        for host in hosts:
            host_is_sshable = _can_connect(host, 22)
            if host_is_sshable:
                hosts.remove(host)
        time.sleep(interval)
        time_elapsed += interval
        if time_elapsed > timeout:
            raise RuntimeError("Couldn't connect to ssh daemon for all hosts. Hosts: " + str(hosts))


def _retry_if_false(result):
    return result is False


def _retry_if_true(result):
    return result is True


@cs.retry(stop_max_delay=30 * 1000,
          wait_fixed=1000,
          retry_on_result=_retry_if_false)
def _wait_for_mpi_to_start_running(master_host):
    return _is_mpi_running(master_host)


@cs.retry(wait_fixed=10000,
          retry_on_result=_retry_if_true)
def _wait_until_mpi_stops_running(master_host):
    return _is_mpi_running(master_host)


def _is_mpi_running(host):
    return _can_connect(host, _PORT)


def _can_connect(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        logger.debug("testing connection to host " + host)
        s.connect((host, port))
        s.close()
        logger.debug("can connect to host " + host)
        return True
    except socket.error:
        logger.debug("can't connect to host " + host)
        return False
