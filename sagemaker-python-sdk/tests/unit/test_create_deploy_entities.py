# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import pytest
from mock import Mock

import sagemaker

MODEL_NAME = 'mymodelname'
ENDPOINT_CONFIG_NAME = 'myendpointconfigname'
ENDPOINT_NAME = 'myendpointname'
ROLE = 'myimrole'
EXPANDED_ROLE = 'arn:aws:iam::111111111111:role/ExpandedRole'
IMAGE = 'myimage'
FULL_CONTAINER_DEF = {'Environment': {}, 'Image': IMAGE, 'ModelDataUrl': 's3://mybucket/mymodel'}
INITIAL_INSTANCE_COUNT = 1
INSTANCE_TYPE = 'ml.c4.xlarge'
REGION = 'us-west-2'


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name='boto_session', region_name=REGION)
    ims = sagemaker.Session(boto_session=boto_mock)
    ims.expand_role = Mock(return_value=EXPANDED_ROLE)
    return ims


def test_create_model(sagemaker_session):

    returned_name = sagemaker_session.create_model(name=MODEL_NAME, role=ROLE, primary_container=FULL_CONTAINER_DEF)

    assert returned_name == MODEL_NAME
    sagemaker_session.sagemaker_client.create_model.assert_called_once_with(
        ModelName=MODEL_NAME,
        PrimaryContainer=FULL_CONTAINER_DEF,
        ExecutionRoleArn=EXPANDED_ROLE)


def test_create_model_expand_primary_container(sagemaker_session):
    sagemaker_session.create_model(name=MODEL_NAME, role=ROLE, primary_container=IMAGE)

    _1, _2, create_model_kwargs = sagemaker_session.sagemaker_client.create_model.mock_calls[0]
    assert create_model_kwargs['PrimaryContainer'] == {'Environment': {}, 'Image': IMAGE}


def test_create_endpoint_config(sagemaker_session):
    returned_name = sagemaker_session.create_endpoint_config(name=ENDPOINT_CONFIG_NAME, model_name=MODEL_NAME,
                                                             initial_instance_count=INITIAL_INSTANCE_COUNT,
                                                             instance_type=INSTANCE_TYPE)

    assert returned_name == ENDPOINT_CONFIG_NAME
    expected_pvs = [{'ModelName': MODEL_NAME,
                     'InitialInstanceCount': INITIAL_INSTANCE_COUNT,
                     'InstanceType': INSTANCE_TYPE,
                     'VariantName': 'AllTraffic'}]
    sagemaker_session.sagemaker_client.create_endpoint_config.assert_called_once_with(
        EndpointConfigName=ENDPOINT_CONFIG_NAME, ProductionVariants=expected_pvs)


def test_create_endpoint_no_wait(sagemaker_session):
    returned_name = sagemaker_session.create_endpoint(
        endpoint_name=ENDPOINT_NAME, config_name=ENDPOINT_CONFIG_NAME, wait=False)

    assert returned_name == ENDPOINT_NAME
    sagemaker_session.sagemaker_client.create_endpoint.assert_called_once_with(
        EndpointName=ENDPOINT_NAME, EndpointConfigName=ENDPOINT_CONFIG_NAME)


def test_create_endpoint_wait(sagemaker_session):
    sagemaker_session.wait_for_endpoint = Mock()
    returned_name = sagemaker_session.create_endpoint(endpoint_name=ENDPOINT_NAME, config_name=ENDPOINT_CONFIG_NAME)

    assert returned_name == ENDPOINT_NAME
    sagemaker_session.sagemaker_client.create_endpoint.assert_called_once_with(
        EndpointName=ENDPOINT_NAME, EndpointConfigName=ENDPOINT_CONFIG_NAME)
    sagemaker_session.wait_for_endpoint.assert_called_once_with(ENDPOINT_NAME)
