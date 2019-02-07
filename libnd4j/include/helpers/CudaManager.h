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
//

#ifndef CUDAMANAGER_H
#define CUDAMANAGER_H

#include <vector>
#include <string>
#include <LaunchContext.h>


namespace nd4j {

class CudaManager {
    
    private:

        nd4j::graph::LaunchContext *_context;
        std::vector<void*> _pOnGlobMem;
        
    public:
        
        CudaManager(nd4j::graph::LaunchContext *context);
        
        ~CudaManager();
        
        void* replicatePointer(const void* src, const size_t size, const std::string& message = "");

        void syncStream(const std::string& message = "") const;
};

}



#endif // CUDAMANAGER_H
