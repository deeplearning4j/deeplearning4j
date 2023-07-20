#!/bin/bash

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

set -exo pipefail
TEST_FILTER="none"




while [[ $# -gt 0 ]]
do
    key="$1"
    value="${2:-}"

    case $key in
        -c|--chip)
        CHIP="${value}"
        shift # past argument
        ;;
        -test|--test-filter)
              TEST_FILTER="${value}"
              shift # past argument
              ;;
        *)
        # unknown option
        ;;
    esac

    if [[ $# -gt 0 ]]; then
        shift # past argument or value
    fi
done

CHIP="${CHIP:-cpu}"
export GTEST_OUTPUT="xml:surefire-reports/TEST-${CHIP}-results.xml"

# On Mac, make sure it can find libraries for GCC
export DYLD_LIBRARY_PATH=/usr/local/lib/gcc/8/:/usr/local/lib/gcc/7/:/usr/local/lib/gcc/6/:/usr/local/lib/gcc/5/

# For Windows, add DLLs of MKL-DNN and OpenBLAS to the PATH
if [ -n "$BUILD_PATH" ]; then
    if which cygpath; then
        BUILD_PATH=$(cygpath -p $BUILD_PATH)
    fi
    export PATH="$PATH:$BUILD_PATH"
fi

# DeclarableOpsTests14.Reshape20
# DeclarableOpsTests14.Reshape17
# DeclarableOpsTests14.Reshape9
unameOut="$(uname)"
echo "$OSTYPE"
if [[ "$TEST_FILTER" != "none" ]]; then
   export BLOCK_SIZE_SCALAR_SCAN=1
   export GRID_SIZE_SCALAR_SCAN=1
   export GRID_SIZE_TRANSFORM_SCAN=1
   export BLOCK_SIZE_TRANSFORM_SCAN=1
   export SHARED_MEM_SIZE_TRANSFORM_SCAN=256
   export GRID_SIZE_COL2IM=256
   export BLOCK_SIZE_COL2IM=256
   export SHARED_MEM_SIZE_COL2IM=16000
    export GRID_SIZE_IM2COL=256
    export BLOCK_SIZE_IM2COL=256
    export SHARED_MEM_SIZE_IM2COL=16000
       export BLOCK_SIZE_RANDOM=128
        export GRID_SIZE_RANDOM=128
  export GRID_SIZE_POOLING=256
  export BLOCK_SIZE_POOLING=256
    export GRID_SIZE_MERGE=256
      export   BLOCK_SIZE_MERGE=256
       export  SHARED_MEM_SIZE_MERGE=256
       export GRID_SIZE_DIAG_PART=128
       export BLOCK_SIZE_DIAG_PART=128
       export GRID_SIZE_SEGMENT_MEAN=128
       export BLOCK_SIZE_SEGMENT_MEAN=128
     export GRID_SIZE_CLIP=128
             export BLOCK_SIZE_CLIP=128
            export GRID_SIZE_SWAP_UNSAFE=128
            export  BLOCK_SIZE_SWAP_UNSAFE=256
            export GRID_SIZE_SEGMENT=128
            export BLOCK_SIZE_SEGMENT=128
            export GRID_SIZE_SEGMENT_MEAN=128
        export BLOCK_SIZE_SEGMENT_MEAN=128
        export GRID_SIZE_GATHER=128
        export BLOCK_SIZE_GATHER=128
         export GRID_SIZE_PREFIX=128
            export BLOCK_SIZE_PREFIX=128
            export GRID_SIZE_ADJUST=128
            export BLOCK_SIZE_ADJUST=128
            export GRID_SIZE_SEGMENT_TAD=128
            export BLOCK_SIZE_SEGMENT_TAD=128
                export GRID_SIZE_MATRIX_DIAG=128
        export BLOCK_SIZE_MATRIX_DIAG=128
       #  export GRID_SIZE_SEGMENT_PROD_2_TAD=128
       # export BLOCK_SIZE_SEGMENT_PROD_2_TAD=128
        export GRID_SIZE_ZETA=64
        export BLOCK_SIZE_ZETA=64
           export GRID_SIZE_SCATTER_SIMPLE=256
   export BLOCK_SIZE_SCATTER_SIMPLE=128
   export GRID_SIZE_MIRROR_PAD_LINEAR=128
   export BLOCK_SIZE_MIRROR_PAD_LINEAR=128
   export GRID_SIZE_POLYGAMMA=64
    export BLOCK_SIZE_POLYGAMMA=64
    export GRID_SIZE_DIGAMMA=128
    export BLOCK_SIZE_DIGAMMA=128
     export GRID_SIZE_BETA_INC=128
       export BLOCK_SIZE_BETA_INC=128
    export GRID_SIZE_INVERT_PERMUTATION=128
    export BLOCK_SIZE_INVERT_PERMUTATION=128


   echo "Running with filter"
   env
  /usr/local/cuda-12.1/bin/compute-sanitizer ../blasbuild/${CHIP}/tests_cpu/layers_tests/runtests --gtest_filter="$TEST_FILTER"

else
  export GRID_SIZE_TRANSFORM_SCAN=1
  export BLOCK_SIZE_TRANSFORM_SCAN=1
  export BLOCK_SIZE_SCALAR_SCAN=1
  export GRID_SIZE_SCALAR_SCAN=1
  export SHARED_MEM_SIZE_TRANSFORM_SCAN=1024
   export GRID_SIZE_COL2IM=128
     export BLOCK_SIZE_COL2IM=128
     export SHARED_MEM_SIZE_COL2IM=16000
  export GRID_SIZE_IM2COL=128
    export BLOCK_SIZE_IM2COL=128
    export SHARED_MEM_SIZE_IM2COL=16000
    export BLOCK_SIZE_RANDOM=128
    export GRID_SIZE_RANDOM=128
    export GRID_SIZE_POOLING=256
      export BLOCK_SIZE_POOLING=256
     export GRID_SIZE_MERGE=256
     export BLOCK_SIZE_MERGE=256
      export SHARED_MEM_SIZE_MERGE=256
          export GRID_SIZE_DIAG_PART=128
             export BLOCK_SIZE_DIAG_PART=128
             export GRID_SIZE_CLIP=128
             export BLOCK_SIZE_CLIP=128
               export GRID_SIZE_SWAP_UNSAFE=128
                         export  BLOCK_SIZE_SWAP_UNSAFE=256
       export GRID_SIZE_SEGMENT_MEAN=128
       export BLOCK_SIZE_SEGMENT_MEAN=128
       export GRID_SIZE_SEGMENT=128
        export BLOCK_SIZE_SEGMENT=128
      export GRID_SIZE_GATHER=128
      export BLOCK_SIZE_GATHER=128
      export GRID_SIZE_PREFIX=128
      export BLOCK_SIZE_PREFIX=128
       export GRID_SIZE_ADJUST=128
        export BLOCK_SIZE_ADJUST=128
        export GRID_SIZE_SEGMENT_TAD=128
        export BLOCK_SIZE_SEGMENT_TAD=128
        export GRID_SIZE_MATRIX_DIAG=128
        export BLOCK_SIZE_MATRIX_DIAG=128
        #export GRID_SIZE_SEGMENT_PROD_2_TAD=128
       # export BLOCK_SIZE_SEGMENT_PROD_2_TAD=128
        export GRID_SIZE_ZETA=64
        export BLOCK_SIZE_ZETA=64
        export GRID_SIZE_SCATTER_SIMPLE=256
        export BLOCK_SIZE_SCATTER_SIMPLE=128
        export GRID_SIZE_MIRROR_PAD_LINEAR=128
        export BLOCK_SIZE_MIRROR_PAD_LINEAR=128
   export GRID_SIZE_DIGAMMA=128
    export BLOCK_SIZE_DIGAMMA=128
       export GRID_SIZE_POLYGAMMA=64
        export BLOCK_SIZE_POLYGAMMA=64
    export GRID_SIZE_ADJUST_WEIGHTS=128
    export BLOCK_SIZE_ADJUST_WEIGHTS=128
    export GRID_SIZE_BETA_INC=128
    export BLOCK_SIZE_BETA_INC=128
        export GRID_SIZE_INVERT_PERMUTATION=128
        export BLOCK_SIZE_INVERT_PERMUTATION=128
  echo "Running without filter"
  env
  valgrind ../blasbuild/${CHIP}/tests_cpu/layers_tests/runtests
fi
# Workaround to fix posix path conversion problem on Windows (http://mingw.org/wiki/Posix_path_conversion)
[ -f "${GTEST_OUTPUT#*:}" ] && cp -a surefire-reports/ ../target && rm -rf surefire-reports/
