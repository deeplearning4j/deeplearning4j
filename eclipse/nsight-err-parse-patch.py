
# Python 3 script to fix error parsing in CMake generated Eclipse project files.
#
# Version 0.01
#
# 2014-12-08 Georg Altmann <georg.altmann@fau.de>

import xml.etree.ElementTree as ET
import sys
import os

def patchCProject(pathToCProject):
    errorParserXml = '<extension id="nvcc.errorParser" point="org.eclipse.cdt.core.ErrorParser"/>'
    tree = ET.parse(pathToCProject)
    root = tree.getroot()
    node = root.find('storageModule/cconfiguration/storageModule/extensions')
    if not node:
        raise RuntimeError('bad eclipse cdt project file: '  + pathToCProject)
    needsPatch = True
    for c in list(node): # iteratre children
        if c.tag == 'extension' and c.attrib['id'] == 'nvcc.errorParser':
            needsPatch = False
            break
    if needsPatch:
        print('patching: ' + pathToCProject)
        patch = ET.fromstring(errorParserXml)
        node.insert(0, patch)
        tree.write(pathToCProject)
    else:
        print('up-to-date: ' + pathToCProject)

    def patchProject(pathToProject):
        tree = ET.parse(pathToProject)
        root = tree.getroot()

        # find error parser list in project file
        node = None
        nextIsVal = False
        for n in root.findall('buildSpec/buildCommand/arguments/dictionary/*'):
            if n.tag == 'key' and n.text == 'org.eclipse.cdt.core.errorOutputParser':
                #print('key '+n.text)
                nextIsVal = True
                continue
            elif nextIsVal and n.tag == 'value':
                #print('val '+n.text)
                node = n
                break

        if None == node:
            raise RuntimeError('did not find errorOutputParser config')

        if node.text.startswith("nvcc.errorParser"):
            print('up-to-date: ' + pathToProject)
        else:
            print('patching: ' + pathToProject)
            # move nvcc.errorParser to the top of the error parser list
            parsers = node.text.split(';')
            if 'nvcc.errorParser' in parsers:
                parsers.remove('nvcc.errorParser')
            parsers.insert(0, 'nvcc.errorParser')
            newParsers = ';'.join(parsers)
            #print('new parsers ' + newParsers)
            node.text = newParsers
            tree.write(pathToProject)
            #print(node.text)

    def usage():
        print('usage:')
        print('nisight-err-parse-patch.py path-to-project')
        print('')
        print('Patches a eclipse project file with nvcc error parsing settings')


        def main():
            if len(sys.argv) < 2:
                usage()
                sys.exit(1)

            path = sys.argv[1]
            if not os.path.isdir(path):
                raise RuntimeError('not a directory: ' + path)

            # It is enough to patch .project, skip .cproject
            #
            # cprojectFilename = os.path.join(path, '.cproject')
            # patchCProject(cprojectFilename)

            projectFilename = os.path.join(path, '.project')
            patchProject(projectFilename)


        if __name__ == '__main__':
            main()