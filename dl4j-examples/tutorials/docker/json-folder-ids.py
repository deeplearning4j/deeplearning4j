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
