
## Nec Aurora


### Vednn


##### Installing vednn

```
#download llvm-ve
wget https://github.com/sx-aurora-dev/llvm-project/releases/download/llvm-ve-rv-v.2.1.0/llvm-ve-rv-2.1-2.1-1.el8.x86_64.rpm
#install llvm-ve
sudo rpm -i llvm-ve-rv-2.1-2.1-1.el8.x86_64.rpm

#find llvm path
LLVM_PATH=$(rpm -ql llvm-ve-rv-2.1-2.1-1.el8.x86_64 | grep lib/cmake/llvm | head -n 1)

#download vednn source files
git clone https://github.com/mergian/vednn
#revert the commit that disabled some kernels
git revert ab93927d71dc6eed5e2c39aad1ae64d2a53e1764
#build and install vednn
cd vednn
mkdir build
cd build
cmake -DLLVM_DIR=${LLVM_PATH} -DCMAKE_INSTALL_PREFIX=/opt/nec/ve ..
make
sudo make install

```

---------------------


### libdn4j  Aurora Ve library with Vednn

##### Building libnd4j with vednn

Specify vednn installation folder if it was not installed in ```/opt/nec/ve```:
```
export VEDNN_ROOT="your installation folder"
```
Specify nlc root:
```
#here we are using 2.3.0
export NLC_ROOT=/opt/nec/ve/nlc/2.3.0/
```
Building libdn4j:
```

./buildnativeoperations.sh -o aurora -dt "float;double;int;int64;bool" -t -h vednn -j 120
```

---------------------

### libdn4j Cpu library with platform  Ve  Vednn acceleration

##### Install VEDA
1st option:

    sudo yum install veoffload-veda.x86_64
2nd option:
install from the source

    git clone https://github.com/SX-Aurora/veda.git
    mkdir -p veda/src/build
    cd veda/src/build
    cmake ..
    make && sudo make install

##### Building libdn4j with Vednn helper:

```
#build libnd4j and tests tuned for the current Cpu with Ve Vednn acceleration
./buildnativeoperations.sh -a native -t -h vednn -j $(nproc)
```
it outputs:
./blasbuild/cpu/libnd4jcpu_so
./blasbuild/cpu/libnd4jcpu_device.so

##### Usage
Veda Ve library should be either in the working directory or it's path should be set using *DEVICE_LIB_LOADPATH* environment variable. If it was build using maven, dl4j/javacpp class loader will set the path to the cached device library.
```
#example:
#running conv2d tests:
export DEVICE_LIB_LOADPATH=${PWD}/blasbuild/cpu/blas
${PWD}/blasbuild/cpu/tests_cpu/layers_tests/runtests --gtest_filter=*conv2*
```
###### Notes
   - vc/vcpp file extensions are used to diferentitate VE device files from c/cpp in cmake. Cmake will compile them using nec.