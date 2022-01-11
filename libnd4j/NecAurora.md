
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

#build and install vednn
cd vednn
mkdir build
cd build
cmake -DLLVM_DIR=${LLVM_PATH} -DCMAKE_INSTALL_PREFIX=/opt/nec/ve  ..
make
sudo make install

```

##### Building libnd4j with vednn

Specify vednn installation folder if it was not installed in ```/opt/nec/ve```:
```
export VEDNN_ROOT="your installation folder"
```

Building libdn4j:
```

./buildnativeoperations.sh -o aurora -dt "float;double;int;int64;bool" -t -h vednn -j 120
```



