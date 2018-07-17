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

import os
import shutil

import simplejson
#from py4j.java_gateway import JavaGateway

input_path_prefix = 'notebook_json/'
output_path_prefix = 'notebook/'

files = [input_path_prefix + file for file in os.listdir("notebook_json")]

if os.path.isdir(output_path_prefix):
    shutil.rmtree(output_path_prefix, ignore_errors=False, onerror=None)
os.mkdir(output_path_prefix)

for file in files:
    file_type = file.split(".")[-1]
    if file_type == 'json' or file_type == 'ipynb':
        with open(file) as data_file:
            if file_type == 'json':
                data = simplejson.load(data_file)

                folder_name = data['id']
                notebook_name = data['name']
                print(folder_name, notebook_name)

                os.mkdir(output_path_prefix + folder_name)
                output_path = output_path_prefix + folder_name + '/note.json'
                data_file.seek(0)

                out = open(output_path,'w')
                out.write(data_file.read())
                out.close()
            elif file_type == 'ipynb':
                print()
                #gateway = JavaGateway()

                #print gateway.entry_point.generateId()
                #print gateway.entry_point.getJson(data_file.read())
