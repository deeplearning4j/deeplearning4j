
## Nec Aurora


### Vednn


##### Prerequisites:

- Veda: version <=  v1.2.0
- Vednn:  https://github.com/mergian/vednn + patch

###### Veda, Llvm-Ve, Vednn installation script

```
#it may require sudo  user for installing veda and llvm-ve
bash libnd4j/build_ve_prerequisites.sh
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


####### Additional notes
If you are facing `Unable to grow stack` error while running the test, then set this environment variable
```
export OMP_STACKSIZE=1G
```
---------------------

### libdn4j Cpu library with platform  Ve  Vednn acceleration

##### Libnd4j + Veda Vednn

```
#build libnd4j and tests tuned for the current Cpu with Ve Vednn acceleration
./buildnativeoperations.sh -a native -t -h vednn -j $(nproc)
```
it outputs:
./blasbuild/cpu/libnd4jcpu_so
./blasbuild/cpu/libnd4jcpu_device.so

##### Usage
Veda Ve library should be either in the working directory or it's path should be set using *DEVICE_LIB_LOADPATH* environment variable.
```
#example:
#running conv2d tests:
export DEVICE_LIB_LOADPATH=${PWD}/blasbuild/cpu/blas
${PWD}/blasbuild/cpu/tests_cpu/layers_tests/runtests --gtest_filter=*conv2*
```

There is not any need to set the device library path manually within nd4j framework. Loading and setting the right device library path are handled inside.

##### Dl4j installation
```
export VEDNN_ROOT="your VEDNN root folder"
helper=vednn
extension=avx2
mvn clean  install  -DskipTests  -Dlibnd4j.helper=${helper} -Dlibnd4j.extension=${extension} -Djavacpp.platform.extension=-${helper}-${extension} -Dlibnd4j.classifier=linux-x86_64-${helper}-${extension} -Pcpu

#or if you followed Prerequisites section
libnd4j/build_veda.sh

```
###### Notes
   - vc/vcpp file extensions are used to diferentitate VE device files from c/cpp in cmake. Cmake will compile them using nec.

