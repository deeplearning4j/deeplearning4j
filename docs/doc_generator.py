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
import re
import os
import shutil
import json
import sys


"""Abstract base class for document generators. Implementations for various programming languages
need to implement the following six methods:

- process_main_docstring
- process_docstring
- render
- get_main_doc_string
- get_constructor_data
- get_public_method_data
"""
class BaseDocumentationGenerator:

    __metaclass__ = abc.ABCMeta

    def __init__(self, args):
        reload(sys)
        sys.setdefaultencoding('utf8')

        self.out_language = args.out_language
        self.template_dir = args.templates if self.out_language == 'en' else args.templates + '_' + self.out_language
        self.project_name = args.project + '/'
        self.validate_templates()

        self.target_dir = args.sources if self.out_language == 'en' else args.sources + '_' + self.out_language
        self.language = args.language
        self.docs_root = args.docs_root
        self.source_code_path = args.code
        self.github_root = ('https://github.com/deeplearning4j/deeplearning4j/tree/master/'
                            + self.source_code_path[3:])

        with open(self.project_name + 'pages.json', 'r') as f:
            json_pages = f.read()
        site = json.loads(json_pages)
        self.pages = site.get('pages', [])
        self.indices = site.get('indices', [])
        self.excludes = site.get('excludes', [])

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


    """Validate language templates
    """
    def validate_templates(self):
        assert os.path.exists(self.project_name + self.template_dir), \
            'No template folder for language ' + self.out_language
        # TODO: check if folder structure for 'templates' and 'templates_XX' aligns
        # TODO: do additional sanity checks to assure different languages are in sync

    """Generate links within documentation.
    """
    def class_to_docs_link(self, module_name, class_name):
        return self.docs_root + module_name.replace('.', '/') + '#' + class_name

    """Generate links to source code.
    """
    def class_to_source_link(self, module_name, cls_name):
        return '[[source]](' + self.github_root + module_name + '/' + cls_name + '.' + self.language + ')'

    """Returns code string as markdown snippet of the respective language.
    """
    def to_code_snippet(self, code):
        return '```' + self.language + '\n' + code + '\n```\n'

    """Returns source code of a class in a module as string.
    """
    def inspect_class_string(self, module, cls):
        return self.read_file(self.source_code_path + module + '/' + cls)

    """Searches for file names within a module to generate an index. The result
    of this is used to create index.md files for each module in question so as
    to easily navigate documentation.
    """
    def read_index_data(self, data):
        module_index = data.get('module_index', "")
        modules = os.listdir(self.project_name + self.target_dir + '/' + module_index)
        modules = [mod.replace('.md', '') for mod in modules if mod != 'index.md']
        index_string = ''.join('- [{}](./{})\n'.format(mod.title().replace('-', ' '), mod) for mod in modules if mod)
        print(index_string)
        return ['', index_string]


    """Grabs page data for each class and allows for iteration in modules and specific classes.
    """
    def organize_page_data(self, module, cls, tag, use_constructors, includes, excludes):
        class_string = self.inspect_class_string(module, cls)
        class_string = self.get_tag_data(class_string, tag)
        class_string = class_string.replace('<p>', '').replace('</p>', '')
        class_name = cls.replace('.' + self.language, '')
        doc_string, class_string = self.get_main_doc_string(class_string, class_name)
        constructors, class_string = self.get_constructor_data(class_string, class_name, use_constructors)
        methods = self.get_public_method_data(class_string, includes, excludes)
        return module, class_name, doc_string, constructors, methods


    """Main workhorse of this script. Inspects source files per class or module and reads
            - class names
            - doc strings of classes / objects
            - doc strings and signatures of methods
            - doc strings and signatures of methods
    Values are returned as nested list, picked up in the main program to write documentation blocks.      
    """
    def read_page_data(self, data):
        if data.get('module_index', ""):  # indices are created after pages
            return []
        page_data = []
        classes = []

        includes = data.get('include', [])
        excludes = data.get('exclude', [])

        use_constructors = data.get('constructors', True)
        tag = data.get('autogen_tag', '')

        modules = data.get('module', "")
        if modules:
            for module in modules:
                module_files = os.listdir(self.source_code_path + module)
                print(module_files)
                for cls in module_files:
                    if '.' in cls:
                        module, class_name, doc_string, constructors, methods = self.organize_page_data(module, cls, tag, use_constructors, includes, excludes)
                        page_data.append([module, class_name, doc_string, constructors, methods])


        class_files = data.get('class', "")
        if class_files:
            for cls in class_files:
                classes.append(cls)

        for cls in sorted(classes):
            module = ""
            module, class_name, doc_string, constructors, methods = self.organize_page_data(module, cls, tag, use_constructors, includes, excludes)
            page_data.append([module, class_name, doc_string, constructors, methods])

        return page_data

    """If a tag is present in a source code string, extract everything between
    tag::<tag>::start and tag::<tag>::end.
    """
    def get_tag_data(self, class_string, tag):
        start_tag = r'tag::' + tag + '::start'
        end_tag = r'tag::' + tag + '::end'
        if not tag:
            return class_string
        elif tag and start_tag in class_string and end_tag not in class_string:
            print("Warning: Start tag, but no end tag found for tag: ", tag)
        elif tag and start_tag in class_string and end_tag not in class_string:
            print("Warning: End tag, but no start tag found for tag: ", tag)
        else:
            start = re.search(start_tag, class_string)
            end = re.search(end_tag, class_string)
            return class_string[start.end():end.start()]

    """Before generating new docs into target folder, clean up old files. 
    """
    def clean_target(self):
        if os.path.exists(self.project_name + self.target_dir):
            shutil.rmtree(self.project_name + self.target_dir)

        for subdir, dirs, file_names in os.walk(self.project_name + self.template_dir):
            for file_name in file_names:
                new_subdir = subdir.replace(self.project_name + self.template_dir, self.project_name + self.target_dir)
                if not os.path.exists(new_subdir):
                    os.makedirs(new_subdir)
                if file_name[-3:] == '.md':
                    file_path = os.path.join(subdir, file_name)
                    new_file_path = self.project_name + self.target_dir + '/' + self.project_name.replace('/','') + '-' + file_name
                    # print(new_file_path)
                    shutil.copy(file_path, new_file_path)


    """Given a file path, read content and return string value.
    """
    def read_file(self, path):
        with open(path) as f:
            return f.read()


    """Create main index.md page for a project by parsing README.md
    and appending it to the template version of index.md
    """
    def create_index_page(self):
        readme = self.read_file(self.project_name + 'README.md')
        index = self.read_file(self.project_name + self.template_dir + '/index.md')
        # if readme has a '##' tag, append it to index
        index = index.replace('{{autogenerated}}', readme[readme.find('##'):])
        with open(self.project_name + self.target_dir + '/index.md', 'w') as f:
            f.write(index)


    """Write blocks of content (arrays of strings) as markdown to
    the file name provided in page_data.
    """
    def write_content(self, blocks, page_data):
        #assert blocks, 'No content for page ' + page_data['page'] # unsure if necessary

        markdown = '\n\n\n'.join(blocks)
        exp_name = self.project_name.replace('/','') + '-' + page_data['page']
        path = os.path.join(self.project_name + self.target_dir, exp_name)

        if os.path.exists(path):
            template = self.read_file(path)
            #assert '{{autogenerated}}' in template, 'Template found for {} but missing {{autogenerated}} tag.'.format(path) # unsure if needed
            markdown = template.replace('{{autogenerated}}', markdown)
        print('Auto-generating docs for {}'.format(path))
        markdown = markdown
        subdir = os.path.dirname(path)
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        with open(path, 'w') as f:
            f.write(markdown)


    """Prepend headers for jekyll, i.e. provide "default" layout and a
    title for the post.
    """
    def prepend_headers(self):
        for subdir, dirs, file_names in os.walk(self.project_name + self.target_dir):
            for file_name in file_names:
                if file_name[-3:] == '.md':
                    file_path = os.path.join(subdir, file_name)
                    header = '---\ntitle: {}\n---\n'.format(file_name.replace('.md', ''))
                    with open(file_path, 'r+') as f:
                        content = f.read()
                        f.seek(0, 0)
                        if not content.startswith('---'):
                            f.write(header.rstrip('\r\n') + '\n' + content)