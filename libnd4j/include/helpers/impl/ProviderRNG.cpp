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

#include <helpers/ProviderRNG.h>
#include <NativeOps.h>


namespace nd4j {
    
ProviderRNG::ProviderRNG() {

    Nd4jLong *buffer = new Nd4jLong[100000];
    NativeOps nativeOps;    
    std::lock_guard<std::mutex> lock(_mutex);
    #ifndef __CUDABLAS__
    // at this moment we don't have streams etc, so let's just skip this for now
    _rng = (nd4j::random::RandomBuffer *) nativeOps.initRandom(nullptr, 123, 100000, (Nd4jPointer) buffer);    
    #endif
    // if(_rng != nullptr)        
}

ProviderRNG& ProviderRNG::getInstance() {     
    
    static ProviderRNG instance; 
    return instance;
}

random::RandomBuffer* ProviderRNG::getRNG() const {

    return _rng;
}

std::mutex ProviderRNG::_mutex;
    
}