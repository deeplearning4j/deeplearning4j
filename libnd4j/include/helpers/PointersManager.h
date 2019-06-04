/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 06.02.2019
// @author raver119@gmail.com
//

#ifndef CUDAMANAGER_H
#define CUDAMANAGER_H

#include <vector>
#include <string>
#include <execution/LaunchContext.h>

#include <types.h>

namespace nd4j {

class PointersManager {

    private:

        nd4j::LaunchContext  *_context;
        std::vector<void*> _pOnGlobMem;
        std::string _funcName;

    public:

        PointersManager(const nd4j::LaunchContext* context, const std::string& funcName = "");

        ~PointersManager();

        void* replicatePointer(const void* src, const size_t size);

        void synchronize() const;

        template<typename T>
        void printDevContentOnHost(const void* pDev, const Nd4jLong len) const;


#ifdef __CUDABLAS__
        template<typename T>
        static void printDevContentOnDevFromHost(const void* pDev, const Nd4jLong len, const int tid = 0);
#endif

#ifdef __CUDACC__
        template<typename T>
        static FORCEINLINE __device__ void printDevContentOnDev(const void* pDev, const Nd4jLong len, const int tid = 0) {
            if(blockIdx.x * blockDim.x + threadIdx.x != tid)
                return;

            printf("device print out: \n");
            for(Nd4jLong i = 0; i < len; ++i)
                printf("%f, ", (double)reinterpret_cast<const T*>(pDev)[i]);

            printf("\n");
        }

#endif

};

}



#endif // CUDAMANAGER_H
