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

import argparse
from java_doc import JavaDocumentationGenerator
from python_doc import PythonDocumentationGenerator
from scala_doc import ScalaDocumentationGenerator

SUPPORTED_LANGUAGES = ["java", "scala", "python"]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--project', '-p', type=str, required=True)  # e.g. keras-import
    parser.add_argument('--code', '-c', type=str, required=True)  # relative path to source code for this project

    parser.add_argument('--language', '-l', type=str, required=False, default='java')
    parser.add_argument('--docs_root', '-d', type=str, required=False, default='http://deeplearning4j.org')
    parser.add_argument('--templates', '-t', type=str, required=False, default='templates')
    parser.add_argument('--sources', '-s', type=str, required=False, default='doc_sources')
    parser.add_argument('--out_language', '-o', type=str, required=False, default='en')

    args = parser.parse_args()

    language = args.language
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError("provided language not supported: {}".format(language))

    if language == "python":
        doc_generator = PythonDocumentationGenerator(args)
    elif language == "scala":
        doc_generator = ScalaDocumentationGenerator(args)
    else:
        doc_generator = JavaDocumentationGenerator(args)

    doc_generator.clean_target()
    #doc_generator.create_index_page() # not necessary for now

    for page_data in doc_generator.pages:
        data = doc_generator.read_page_data(page_data)
        blocks = []
        for module_name, class_name, doc_string, constructors, methods in data:
            class_string = doc_generator.inspect_class_string(module_name, class_name + '.' + doc_generator.language)
            # skip class if it contains any exclude keywords
            if not any(ex in class_string for ex in doc_generator.excludes):
                sub_blocks = []
                link = doc_generator.class_to_source_link(module_name, class_name)
                try:
                    class_name = class_name.rsplit('/',1)[1]
                except:
                    print('Skipping split on '+class_name)
                # if module_name:
                #     sub_blocks.append('### {}'.format(module_name))
                #     sub_blocks.append('<span style="float:right;"> {} </span>\n'.format(link))

                if doc_string:
                    sub_blocks.append('\n---\n')
                    sub_blocks.append('### {}'.format(class_name))
                    sub_blocks.append('<span style="float:right;"> {} </span>\n'.format(link))
                    sub_blocks.append(doc_string)

                if constructors:
                    sub_blocks.append("".join([doc_generator.render(cs, cd, class_name, False) for (cs, cd) in constructors]))

                if methods:
                    # sub_blocks.append('<button class="btn btn-primary" type="button" data-toggle="collapse" data-target="#'+class_name+'" aria-expanded="false" aria-controls="'+class_name+'">Show methods</button>')
                    # sub_blocks.append('<div class="collapse" id="'+class_name+'"><div class="card card-body">\n')
                    sub_blocks.append("".join([doc_generator.render(ms, md, class_name, True) for (ms, md) in methods]))
                    # sub_blocks.append('</div></div>')                   
                blocks.append('\n'.join(sub_blocks))

        doc_generator.write_content(blocks, page_data)

    for index_data in doc_generator.indices:
        index = doc_generator.read_index_data(index_data)
        doc_generator.write_content(index, index_data)

doc_generator.prepend_headers()
