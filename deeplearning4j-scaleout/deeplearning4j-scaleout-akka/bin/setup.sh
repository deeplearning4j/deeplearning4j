#!/bin/bash
RED_HAT=1
DEBIAN=2
OS=0

find_os() {
  
   if [ -f "/etc/redhat-release" ]; then
      OS=RED_HAT
   elif [ -f "/etc/issue"]; then
      OS=DEBIAN
   fi
}



echo "$OS"


install_python() {
 sudo pip install matplotlib scipy numpy supervisor

}



install_setup() {
find_os

if [[("$OS"=="$REDHAT")]]; then
    echo "Install Red hat..."
    sudo yum -y install blas java-1.7.0-openjdk-devel.x86_64 python-setuptools numpy scipy python-matplotlib ipython python-pandas sympy python-nose
    sudo easy_install pip
    install_python

elif [[("$OS" == "$DEBIAN")]]; then 
    echo "Installing debian..."
    apt-get install blas openjdk-7-jdk python-pip python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
    install_python
fi

}

install_setup


