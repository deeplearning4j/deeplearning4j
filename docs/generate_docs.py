# -*- coding: utf-8 -*-
import re
import os
import shutil
import json
import argparse


class DocumentationGenerator:

    def __init__(self, args):
        self.template_dir = args.templates
        self.target_dir = args.sources
        self.project_name = args.project + '/'
        self.language = args.language
        self.docs_root = args.docs_root
        self.source_code_path = args.code
        self.github_root = ('https://github.com/deeplearning4j/deeplearning4j/tree/master/'
                            + self.source_code_path[3:])


    def class_to_docs_link(self, module_name, class_name):
        return self.docs_root + module_name.replace('.', '/') + '#' + class_name


    def class_to_source_link(self, module_name, cls_name):
        return '[[source]](' + self.github_root + module_name + '/' + cls_name + '.' + self.language + ')'


    def to_code_snippet(self, code):
        return '```' + self.language + '\n' + code + '\n```\n'


    def process_main_docstring(self, doc_string):
        lines = doc_string.split('\n')
        doc = [line.replace('*', '').lstrip(' ').rstrip('/') for line in lines[1:-1] if not '@' in line]
        return '\n'.join(doc)


    def process_docstring(self, doc_string):
        lines = doc_string.split('\n')
        doc = [line.replace('*', '').lstrip(' ').replace('@', '- ') for line in lines]
        return '\n'.join(doc)


    def render(self, signature, doc_string):
        name = signature
        subblocks = ['<b<{}</b> \n{}'.format(name, self.to_code_snippet(signature))]
        if doc_string:
            subblocks.append(doc_string + '\n')
        return '\n\n'.join(subblocks)


    def get_main_doc_string(self, class_string, class_name):
        doc_regex = r'\/\*\*\n([\S\s]*?.*)\*\/\n'  # match "/** ... */" at the top
        doc_string = re.search(doc_regex, class_string)
        doc = self.process_main_docstring(doc_string.group())
        if not doc_string:
            print('Warning, no doc string found for class {}'.format(class_name))
        return doc, class_string[doc_string.end():]


    def get_constructor_data(self, class_string, class_name):
        constructors = []
        if 'public ' + class_name in class_string:
            while 'public ' + class_name in class_string:
                doc_regex = r'\/\*\*\n([\S\s]*?.*)\*\/\n[\S\s]*?(public ' \
                            + class_name + '.[\S\s]*?){'
                result = re.search(doc_regex, class_string)
                doc_string, signature = result.groups()
                doc = self.process_docstring(doc_string)
                class_string = class_string[result.end():]
                constructors.append((signature, doc))
        return constructors, class_string

    def get_public_method_data(self, class_string):
        method_regex = r'public [static\s]?[a-zA-Z0-9]* ([a-zA-Z0-9]*)\('
        method_strings = re.findall(method_regex, class_string)

        methods = []
        for method in method_strings:
            doc_regex = r'\/\*\*\n([\S\s]*?.*)\*\/\n[\S\s]*?' + \
                        '(public [static\s]?[a-zA-Z0-9]*.[\S\s]*?){'
            result = re.search(doc_regex, class_string)
            doc_string, signature = result.groups()
            doc = self.process_docstring(doc_string)
            class_string = class_string[result.end():]
            methods.append((signature, doc))
        return methods


    def read_page_data(self, data):
        classes = []
        module = data.get('module', "")
        if module:
            classes = os.listdir(self.source_code_path + module)
        cls = data.get('class', "")
        if cls:
            classes = cls
        page_data = []
        for c in classes:
            class_string = self.read_file(self.source_code_path + module + '/' + c)
            class_name = c.strip('.' + self.language)
            doc_string, class_string = self.get_main_doc_string(class_string, class_name)
            constructors, class_string = self.get_constructor_data(class_string, class_name)
            methods = self.get_public_method_data(class_string)

            page_data.append([module, class_name, doc_string, constructors, methods])

        return page_data


    def clean_target(self):
        if os.path.exists(self.project_name + self.target_dir):
            shutil.rmtree(self.project_name + self.target_dir)

        for subdir, dirs, fnames in os.walk(self.project_name + self.template_dir):
            for fname in fnames:
                new_subdir = subdir.replace(self.project_name + self.template_dir, self.project_name + self.target_dir)
                if not os.path.exists(new_subdir):
                    os.makedirs(new_subdir)
                if fname[-3:] == '.md':
                    fpath = os.path.join(subdir, fname)
                    new_fpath = fpath.replace(self.project_name + self.template_dir, self.project_name + self.target_dir)
                    shutil.copy(fpath, new_fpath)



    def read_file(self, path):
        with open(path) as f:
            return f.read()


    def create_index_page(self):
        readme = self.read_file(self.project_name + 'README.md')
        index = self.read_file(self.project_name + 'templates/index.md')
        index = index.replace('{{autogenerated}}', readme[readme.find('##'):])
        with open(self.project_name + self.target_dir + '/index.md', 'w') as f:
            f.write(index)

    def write_content(self, blocks, page_data):
        assert blocks, 'No content for page ' + page_data['page']

        markdown = '\n----\n\n'.join(blocks)
        path = os.path.join(self.project_name + self.target_dir, page_data['page'])
        if os.path.exists(path):
            template = self.read_file(path)
            assert '{{autogenerated}}' in template, \
                    'Template found for {} but missing {{autogenerated}} tag.'.format(path)
            markdown = template.replace('{{autogenerated}}', markdown)
        print('Auto-generating docs for {}'.format(path))

        subdir = os.path.dirname(path)
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        with open(path, 'w') as f:
            f.write(markdown)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--project', '-p', type=str, required=True)  # e.g. keras-import
    parser.add_argument('--code', '-c', type=str, required=True)  # relative path to source code for this project

    parser.add_argument('--language', '-l', type=str, required=False, default='java')
    parser.add_argument('--docs_root', '-d', type=str, required=False, default='http://deeplearning4j.org') # TBD
    parser.add_argument('--templates', '-t', type=str, required=False, default='templates')
    parser.add_argument('--sources', '-s', type=str, required=False, default='doc_sources')

    args = parser.parse_args()
    doc_generator = DocumentationGenerator(args)

    doc_generator.clean_target()
    doc_generator.create_index_page()

    with open(doc_generator.project_name + 'pages.json', 'r') as f:
        json_pages = f.read()
    pages = json.loads(json_pages)

    for page_data in pages:
        data = doc_generator.read_page_data(page_data)
        blocks = []
        for module_name, class_name, doc_string, constructors, methods in data:
            subblocks = []
            link = doc_generator.class_to_source_link(module_name, class_name)
            subblocks.append('<span style="float:right;"> {} </span>'.format(link))
            if module_name:
                subblocks.append('## {}\n'.format(class_name))

            if doc_string:
                subblocks.append(doc_string)

            if constructors:
                subblocks.append('\n---')
                subblocks.append('<b>Constructors</b>\n')
                subblocks.append('\n---\n'.join([doc_generator.render(cs, cd) for (cs, cd) in constructors]))

            if methods:
                subblocks.append('\n---')
                subblocks.append('<b>Methods</b>\n')
                subblocks.append('\n---\n'.join([doc_generator.render(ms, md) for (ms, md) in methods]))
            blocks.append('\n'.join(subblocks))

        doc_generator.write_content(blocks, page_data)

