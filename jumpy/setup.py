################################################################################
# Copyright (c) 2015-2018 Skymind, Inc.
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# SPDX-License-Identifier: Apache-2.0
################################################################################


from setuptools import setup
from setuptools import find_packages


setup(name='jumpy',
      version='0.2.4',
      description='Numpy and nd4j interop',
      long_description='Mapping of the numpy & nd4j array representations',
      author='Adam Gibson',
      author_email='adam@skymind.io',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3',
          'Topic :: Software Development :: Libraries',
      ],
      keywords='numpy jumpy java nd4j deeplearning4j',
      url='https://github.com/deeplearning4j/deeplearning4j.git',
      license='Apache',
      setup_requires=['Cython', 'pytest-runner'],
      install_requires=['Cython', 'pyjnius', 'numpy'],
      extras_require={
          'visualize': ['pydot-ng'],
          'tests': ['pytest',
                    'pytest-pep8',
                    'mock'],
      },
      packages=find_packages())
