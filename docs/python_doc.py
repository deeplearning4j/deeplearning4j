# -*- coding: utf-8 -*-

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

import abc
import sys
from doc_generator import BaseDocumentationGenerator


class PythonDocumentationGenerator(BaseDocumentationGenerator):

    def __init__(self, args):
        reload(sys)
        sys.setdefaultencoding('utf8')

        super(PythonDocumentationGenerator, self).__init__(args)

        raise NotImplementedError

    """Process top class docstring
    """
    @abc.abstractmethod
    def process_main_docstring(self, doc_string):
        raise NotImplementedError

    """Process method and other docstrings
    """
    @abc.abstractmethod
    def process_docstring(self, doc_string):
        raise NotImplementedError

    """Takes unformatted signatures and doc strings and returns a properly
    rendered piece that fits into our markdown layout.
    """
    @abc.abstractmethod
    def render(self, signature, doc_string, class_name, is_method):
        raise NotImplementedError


    """Returns main doc string of class/object in question.
    """
    @abc.abstractmethod
    def get_main_doc_string(self, class_string, class_name):
        raise NotImplementedError


    """Returns doc string and signature data for constructors.
    """
    @abc.abstractmethod
    def get_constructor_data(self, class_string, class_name, use_contructor):
        raise NotImplementedError


    """Returns doc string and signature data for methods
    in the public API of an object
    """
    @abc.abstractmethod
    def get_public_method_data(self, class_string, includes, excludes):
        raise NotImplementedError

