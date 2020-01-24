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
#include <vector>
#include <dll.h>
#include <stdexcept>
#include <array/DataType.h>
#include <types/pair.h>
#include <pointercast.h>

namespace nd4j{
    class ND4J_EXPORT Environment {
    private:
        std::atomic<int> _tadThreshold;
        std::atomic<int> _elementThreshold;
        std::atomic<bool> _verbose;
        std::atomic<bool> _debug;
        std::atomic<bool> _leaks;
        std::atomic<bool> _profile;
        std::atomic<nd4j::DataType> _dataType;
        std::atomic<bool> _precBoost;
        std::atomic<bool> _useMKLDNN{true};
        std::atomic<bool> _allowHelpers{true};

        std::atomic<int> _maxThreads;
        std::atomic<int> _maxMasterThreads;

        // these fields hold defaults
        std::atomic<int64_t> _maxTotalPrimaryMemory{-1};
        std::atomic<int64_t> _maxTotalSpecialMemory{-1};
        std::atomic<int64_t> _maxDeviceMemory{-1};

#ifdef __ND4J_EXPERIMENTAL__
        const bool _experimental = true;
#else
        const bool _experimental = false;
#endif

        // device compute capability for CUDA
        std::vector<Pair> _capabilities;

        static Environment* _instance;

        Environment();
        ~Environment();
    public:
        /**
         * These 3 fields are mostly for CUDA/cuBLAS version tracking
         */
        int _blasMajorVersion = 0;
        int _blasMinorVersion = 0;
        int _blasPatchVersion = 0;

        static Environment* getInstance();

        bool isVerbose();
        void setVerbose(bool reallyVerbose);
        bool isDebug();
        bool isProfiling();
        bool isDetectingLeaks();
        bool isDebugAndVerbose();
        void setDebug(bool reallyDebug);
        void setProfiling(bool reallyProfile);
        void setLeaksDetector(bool reallyDetect);
        bool helpersAllowed();
        void allowHelpers(bool reallyAllow);
        
        int tadThreshold();
        void setTadThreshold(int threshold);

        int elementwiseThreshold();
        void setElementwiseThreshold(int threshold);

        int maxThreads();
        void setMaxThreads(int max);

        int maxMasterThreads();
        void setMaxMasterThreads(int max);

        /*
         * Legacy memory limits API, still used in new API as simplified version
         */
        void setMaxPrimaryMemory(uint64_t maxBytes);
        void setMaxSpecialyMemory(uint64_t maxBytes);
        void setMaxDeviceMemory(uint64_t maxBytes);

        uint64_t maxPrimaryMemory();
        uint64_t maxSpecialMemory();
        ////////////////////////

        /*
         * Methods for memory limits/counters
         */
        void setGroupLimit(int group, Nd4jLong numBytes);
        void setDeviceLimit(int deviceId, Nd4jLong numBytes);

        Nd4jLong getGroupLimit(int group);
        Nd4jLong getDeviceLimit(int deviceId);

        Nd4jLong getGroupCounter(int group);
        Nd4jLong  getDeviceCounter(int deviceId);
        ////////////////////////

        bool isUseMKLDNN() { return _useMKLDNN.load(); }
        void setUseMKLDNN(bool useMKLDNN) { _useMKLDNN.store(useMKLDNN); }

        nd4j::DataType defaultFloatDataType();
        void setDefaultFloatDataType(nd4j::DataType dtype);

        bool precisionBoostAllowed();
        void allowPrecisionBoost(bool reallyAllow);

        bool isExperimentalBuild();

        bool isCPU();

        int blasMajorVersion();
        int blasMinorVersion();
        int blasPatchVersion();

        std::vector<Pair>& capabilities();
    };
}


#endif //LIBND4J_ENVIRONMENT_H
