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

import re
import sys
from doc_generator import BaseDocumentationGenerator


class JavaDocumentationGenerator(BaseDocumentationGenerator):

    def __init__(self, args):
        reload(sys)
        sys.setdefaultencoding('utf8')

        super(JavaDocumentationGenerator, self).__init__(args)

    '''Doc strings (in Java/Scala) need to be stripped of all '*' values.
    Convert '@param' to '- param'. Strip line with author as well.
    
    TODO can be vastly improved.
    '''
    def process_main_docstring(self, doc_string):
        lines = doc_string.split('\n')
        doc = [line.replace('*', '').lstrip(' ').rstrip('/') for line in lines[1:-1] if not '@' in line]
        return '\n'.join(doc)


    '''Doc strings (in Java/Scala) need to be stripped of all '*' values.
    Convert '@param' to '- param'. TODO can be vastly improved.
    '''
    def process_docstring(self, doc_string):
        lines = doc_string.split('\n')
        doc = [line.replace('*', '').lstrip(' ').replace('@', '- ') for line in lines]
        return '\n'.join(doc)


    '''Takes unformatted signatures and doc strings and returns a properly
    rendered piece that fits into our markdown layout.
    '''
    def render(self, signature, doc_string, class_name, is_method):
        if is_method:  # Method name from signature
            method_regex = r'public (?:static )?[a-zA-Z0-9]* ([a-zA-Z0-9]*)\('
            name = re.findall(method_regex, signature)[0]
        else:  # Constructor takes class name
            name = class_name
        sub_blocks = ['##### {} \n{}'.format(name, self.to_code_snippet(signature))]
        if doc_string:
            sub_blocks.append(doc_string + '\n')
        return '\n\n'.join(sub_blocks)


    '''Returns main doc string of class/object in question.
    '''
    def get_main_doc_string(self, class_string, class_name):
        print(class_name)
        doc_regex = r'\/\*\*\n([\S\s]*?.*)\*\/\n'  # match "/** ... */" at the top
        doc_string = re.search(doc_regex, class_string)
        try:
            doc_match = doc_string.group();
        except:
            doc_match = ''
        doc = self.process_main_docstring(doc_match)
        if not doc_string:
            print('Warning, no doc string found for class {}'.format(class_name))
        doc_index = 0 if not doc_match else doc_string.end()
        return doc, class_string[doc_index:]


    '''Returns doc string and signature data for constructors.
    '''
    def get_constructor_data(self, class_string, class_name, use_contructor):
        constructors = []
        if 'public ' + class_name in class_string and use_contructor:
            doc_regex = r'\/\*\*\n([\S\s]*?.*)\*\/\n[\S\s]*?(public ' \
                        + class_name + '.[\S\s]*?){'
            result = re.search(doc_regex, class_string)
            if result:
                doc_string, signature = result.groups()
                doc = self.process_docstring(doc_string)
                class_string = class_string[result.end():]
                constructors.append((signature, doc))
            else:
                print("Warning, no doc string found for constructor {}".format(class_name))
        return constructors, class_string


    '''Returns doc string and signature data for methods
    in the public API of an object
    '''
    def get_public_method_data(self, class_string, includes, excludes):
        method_regex = r'public (?:static )?[a-zA-Z0-9]* ([a-zA-Z0-9]*)\('

        # Either use all methods or use include methods that can be found
        method_strings = re.findall(method_regex, class_string)
        if includes:
            method_strings = [i for i in includes if i in method_strings]

        # Exclude all 'exclude' methods
        method_strings = [m for m in method_strings if m not in excludes]

        methods = []
        for method in method_strings:
            # print("Processing doc string for method {}".format(method))
            doc_regex = r'\/\*\*\n([\S\s]*?.*)\*\/\n[\S\s]*?' + \
                        '(public (?:static )?[a-zA-Z0-9]* ' + method + '[\S\s]*?){'
            # TODO: this will sometimes run forever. fix regex
            result = re.search(doc_regex, class_string)
            if result:
                doc_string, signature = result.groups()
                doc = self.process_docstring(doc_string)
                class_string = class_string[result.end():]
                methods.append((signature, doc))
            else:
                print("Warning, no doc string found for method {}".format(method))
        return methods
