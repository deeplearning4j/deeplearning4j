import os

filepath = os.path.dirname(os.path.abspath(__file__))
pom = filepath + '/nd4j-backends/nd4j-backend-impls/nd4j-cuda/pom.xml'

def get_cuda_version():
    with open(pom) as f:
        txt = f.read()
    cuda_version = txt.split('<artifactId>nd4j-cuda-')[1].split('</artifactId>')[0]
    return cuda_version


cuda_version = get_cuda_version()
print('cuda version : ' + cuda_version)

build_command = 'mvn clean install -X -D skipTests -D maven.javadoc.skip=true -pl "!:nd4j-tests,!:nd4j-tensorflow" -rf :nd4j-native'

build_command = build_command.format(cuda_version, cuda_version)

os.system(build_command)