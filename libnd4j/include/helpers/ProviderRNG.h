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
// Created by Yurii Shyrma on 27.01.2018
//

#ifndef LIBND4J_PROVIDERRNG_H
#define LIBND4J_PROVIDERRNG_H

#include <helpers/helper_random.h>
#include <mutex>

namespace nd4j {
    
class ProviderRNG {
        
    protected:
        random::RandomBuffer* _rng;
        static std::mutex _mutex;
        ProviderRNG();

    public:
        ProviderRNG(const ProviderRNG&)    = delete;
        void operator=(const ProviderRNG&) = delete;   
        random::RandomBuffer* getRNG() const;
        static ProviderRNG& getInstance();        
};


}

#endif //LIBND4J_PROVIDERRNG_H
