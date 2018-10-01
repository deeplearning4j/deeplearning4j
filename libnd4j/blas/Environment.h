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
// Created by raver119 on 06.10.2017.
//

#ifndef LIBND4J_ENVIRONMENT_H
#define LIBND4J_ENVIRONMENT_H

#include <atomic>
#include <dll.h>
#include <helpers/StringUtils.h>
#include <stdexcept>
#include <array/DataType.h>

namespace nd4j{
    class ND4J_EXPORT Environment {
    private:
        std::atomic<int> _tadThreshold;
        std::atomic<int> _elementThreshold;
        std::atomic<bool> _verbose;
        std::atomic<bool> _debug;
        std::atomic<bool> _profile;
        std::atomic<int> _maxThreads;
        std::atomic<nd4j::DataType> _dataType;
        std::atomic<bool> _precBoost;

        static Environment* _instance;

        Environment();
        ~Environment();
    public:
        static Environment* getInstance();

        bool isVerbose();
        void setVerbose(bool reallyVerbose);
        bool isDebug();
        bool isProfiling();
        bool isDebugAndVerbose();
        void setDebug(bool reallyDebug);
        void setProfiling(bool reallyProfile);
        
        int tadThreshold();
        void setTadThreshold(int threshold);

        int elementwiseThreshold();
        void setElementwiseThreshold(int threshold);

        int maxThreads();
        void setMaxThreads(int max);

        nd4j::DataType defaultFloatDataType();
        void setDefaultFloatDataType(nd4j::DataType dtype);

        bool precisionBoostAllowed();
        void allowPrecisionBoost(bool reallyAllow);
    };
}


#endif //LIBND4J_ENVIRONMENT_H
