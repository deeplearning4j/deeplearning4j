import simplejson
import shutil
import os, sys

input_path_prefix = 'notebook_json/'
output_path_prefix = 'notebook/'

files = [input_path_prefix + file for file in os.listdir("notebook_json")]

shutil.rmtree(output_path_prefix, ignore_errors=False, onerror=None)
os.mkdir(output_path_prefix)
for file in files:
    with open(file) as data_file:
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