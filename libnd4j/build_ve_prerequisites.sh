#!/usr/bin/env bash
#
# /* ******************************************************************************
#  *
#  *
#  * This program and the accompanying materials are made available under the
#  * terms of the Apache License, Version 2.0 which is available at
#  * https://www.apache.org/licenses/LICENSE-2.0.
#  *
#  *  See the NOTICE file distributed with this work for additional
#  *  information regarding copyright ownership.
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#  * License for the specific language governing permissions and limitations
#  * under the License.
#  *
#  * SPDX-License-Identifier: Apache-2.0
#  ******************************************************************************/
#

function message {
    echo ":::: ${@}"
}

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
BASE_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"


cd ${BASE_DIR}
isVeda=$(rpm -q veoffload-veda.x86_64 | grep -o "is not installed")
if [[ $isVeda == "is not installed" ]] ; then

[ "$UID" -eq 0 ] || { message "This script must be run as root or with sudo to install Veda."; exit 1;}
message "install Veda"
sudo yum install veoffload-veda.x86_64 -y

fi

export VEDNN_ROOT=${BASE_DIR}/vednn_lib
if [ ! -f "${VEDNN_ROOT}/lib/libvednn_openmp.a" ]; then
message "build Vednn"

isLLVMVE=$(rpm -q  llvm-ve-rv-2.1-2.1-1.el8.x86_64.rpm | grep -o "is not installed" )
if [[ $isLLVMVE == "is not installed" ]] ; then
[ "$UID" -eq 0 ] || { message "This script must be run as root or with sudo to install Veda."; exit 1;}
message "download llvm-ve"
wget -q --show-progress https://github.com/sx-aurora-dev/llvm-project/releases/download/llvm-ve-rv-v.2.1.0/llvm-ve-rv-2.1-2.1-1.el8.x86_64.rpm
message "install llvm-ve"
sudo rpm -i llvm-ve-rv-2.1-2.1-1.el8.x86_64.rpm
fi
#find llvm path
LLVM_PATH=$(rpm -ql llvm-ve-rv-2.1-2.1-1.el8.x86_64 | grep lib/cmake/llvm | head -n 1)

#instal dir in VEDNN_ROOT

mkdir -p ${VEDNN_ROOT}
message "download and install Vednn"
#download vednn source files
git clone https://github.com/mergian/vednn
#build and install vednn
git clone https://github.com/mergian/vednn
cd vednn
git checkout f311ed1c57635e19e4f3acd36e087121dcf89d8c
git apply ../vednn_mergian.patch
mkdir build
cd build
cmake -DLLVM_DIR=${LLVM_PATH} -DCMAKE_INSTALL_PREFIX=${VEDNN_ROOT} ..
make
make install

fi

