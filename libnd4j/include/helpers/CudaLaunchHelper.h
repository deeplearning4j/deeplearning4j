/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// Created by raver on 4/5/2018.
//

#ifndef LIBND4J_CUDALAUNCHHELPER_H
#define LIBND4J_CUDALAUNCHHELPER_H


#include <system/pointercast.h>
#include <system/dll.h>
#include <system/op_boilerplate.h>
#include <types/triple.h>

namespace sd {
    class ND4J_EXPORT CudaLaunchHelper {
    public:
        static Triple getFlatLaunchParams(Nd4jLong length, int SM, int CORES, int SHARED_MEMORY);
        static int getReductionBlocks(Nd4jLong xLength, int blockSize = 512);
    };
}


#endif //LIBND4J_CUDALAUNCHHELPER_H
