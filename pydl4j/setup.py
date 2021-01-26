#  /* ******************************************************************************
#   * Copyright (c) 2021 Deeplearning4j Contributors
#   *
#   * This program and the accompanying materials are made available under the
#   * terms of the Apache License, Version 2.0 which is available at
#   * https://www.apache.org/licenses/LICENSE-2.0.
#   *
#   * Unless required by applicable law or agreed to in writing, software
#   * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#   * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#   * License for the specific language governing permissions and limitations
#   * under the License.
#   *
#   * SPDX-License-Identifier: Apache-2.0
#   ******************************************************************************/

################################################################################
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

setup(
    name='pydl4j',
    version='0.1.5',
    packages=find_packages(),
    install_requires=['Cython', 'pyjnius', 'requests',
                      'click', 'argcomplete', 'python-dateutil'],
    extras_require={
        'tests': ['pytest', 'pytest-pep8', 'pytest-cov']
    },
    include_package_data=True,
    license='Apache',
    description='Java dependency management for Python projects using DL4J',
    url='https://github.com/deeplearning4j/pydl4j',
    entry_points={
        'console_scripts': [
            'pydl4j=pydl4j.cli:handle'
        ]
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Environment :: Console',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3'
    ]
)
