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

#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include "system/Environment.h"
#include <helpers/StringUtils.h>
#include <thread>
#include <helpers/logger.h>
#include <memory/MemoryCounter.h>

#ifdef _OPENMP

#include <omp.h>

#endif

#ifdef __CUDABLAS__

#include <cuda.h>
#include <cuda_runtime.h>
#include <system/BlasVersionHelper.h>
#endif

namespace sd {

    sd::Environment::Environment() {
        _tadThreshold.store(1);
        _elementThreshold.store(1024);
        _verbose.store(false);
        _debug.store(false);
        _profile.store(false);
        _precBoost.store(false);
        _leaks.store(false);
        _dataType.store(sd::DataType::FLOAT32);
        _maxThreads = std::thread::hardware_concurrency();
        _maxMasterThreads = _maxThreads.load();

#ifndef ANDROID
        const char* omp_threads = std::getenv("OMP_NUM_THREADS");
        if (omp_threads != nullptr) {
            try {
                std::string omp(omp_threads);
                int val = std::stoi(omp);
                _maxThreads.store(val);
                _maxMasterThreads.store(val);
            } catch (std::invalid_argument &e) {
                // just do nothing
            } catch (std::out_of_range &e) {
                // still do nothing
            }
        }

        /**
         * Defines size of thread pool used for parallelism
         */
        const char* max_threads = std::getenv("SD_MAX_THREADS");
        if (max_threads != nullptr) {
            try {
                std::string t(max_threads);
                int val = std::stoi(t);
                _maxThreads.store(val);
            } catch (std::invalid_argument &e) {
                // just do nothing
            } catch (std::out_of_range &e) {
                // still do nothing
            }
        }

        /**
         * Defines max number of threads usable at once
         */
        const char* max_master_threads = std::getenv("SD_MASTER_THREADS");
        if (max_master_threads != nullptr) {
            try {
                std::string t(max_master_threads);
                int val = std::stoi(t);
                _maxMasterThreads.store(val);
            } catch (std::invalid_argument &e) {
                // just do nothing
            } catch (std::out_of_range &e) {
                // still do nothing
            }
        }

        if (_maxMasterThreads.load() > _maxThreads.load()) {
            nd4j_printf("Warning! MAX_MASTER_THREADS > MAX_THREADS, tuning them down to match each other\n","");
            _maxMasterThreads.store(_maxThreads.load());
        }

        /**
         * If this env var is defined - we'll disallow use of platform-specific helpers (mkldnn, cudnn, etc)
         */
        const char* forbid_helpers = std::getenv("SD_FORBID_HELPERS");
        if (max_master_threads != nullptr) {
            _allowHelpers = false;
        }

        /**
         * This var defines max amount of host memory library can allocate
         */
        const char* max_primary_memory = std::getenv("SD_MAX_PRIMARY_BYTES");
        if (max_primary_memory != nullptr) {
            try {
                std::string t(max_primary_memory);
                auto val = std::stol(t);
                _maxTotalPrimaryMemory.store(val);
            } catch (std::invalid_argument &e) {
                // just do nothing
            } catch (std::out_of_range &e) {
                // still do nothing
            }
        }

        /**
         * This var defines max amount of special (i.e. device) memory library can allocate on all devices combined
         */
        const char* max_special_memory = std::getenv("SD_MAX_SPECIAL_BYTES");
        if (max_special_memory != nullptr) {
            try {
                std::string t(max_special_memory);
                auto val = std::stol(t);
                _maxTotalSpecialMemory.store(val);
            } catch (std::invalid_argument &e) {
                // just do nothing
            } catch (std::out_of_range &e) {
                // still do nothing
            }
        }

        /**
         * This var defines max amount of special (i.e. device) memory library can allocate on all devices combined
         */
        const char* max_device_memory = std::getenv("SD_MAX_DEVICE_BYTES");
        if (max_device_memory != nullptr) {
            try {
                std::string t(max_device_memory);
                auto val = std::stol(t);
                _maxDeviceMemory.store(val);
            } catch (std::invalid_argument &e) {
                // just do nothing
            } catch (std::out_of_range &e) {
                // still do nothing
            }
        }

        const char* blas_fallback = std::getenv("SD_BLAS_FALLBACK");
        if (blas_fallback != nullptr) {
            _blasFallback = true;
        }
#endif

#ifdef __CUDABLAS__
        int devCnt = 0;
	    cudaGetDeviceCount(&devCnt);
	    auto devProperties = new cudaDeviceProp[devCnt];
	    for (int i = 0; i < devCnt; i++) {
		    cudaSetDevice(i);
		    cudaGetDeviceProperties(&devProperties[i], i);

		    //cudaDeviceSetLimit(cudaLimitStackSize, 4096);
		    Pair p(devProperties[i].major, devProperties[i].minor);
		    _capabilities.emplace_back(p);
	    }

	    BlasVersionHelper ver;
        _blasMajorVersion = ver._blasMajorVersion;
        _blasMinorVersion = ver._blasMinorVersion;
        _blasPatchVersion = ver._blasPatchVersion;

	    cudaSetDevice(0);
	    delete[] devProperties;
#else

#endif
    }

    bool sd::Environment::blasFallback() {
        return _blasFallback;
    }

    sd::Environment::~Environment() {
        //
    }

    void Environment::setMaxPrimaryMemory(uint64_t maxBytes) {
        _maxTotalPrimaryMemory = maxBytes;
    }

    void Environment::setMaxSpecialyMemory(uint64_t maxBytes) {
        _maxTotalSpecialMemory = maxBytes;
    }

    void Environment::setMaxDeviceMemory(uint64_t maxBytes) {
        _maxDeviceMemory = maxBytes;
    }

    Environment *Environment::getInstance() {
        if (_instance == 0)
            _instance = new Environment();

        return _instance;
    }

    bool Environment::isVerbose() {
        return _verbose.load();
    }

    bool Environment::isExperimentalBuild() {
        return _experimental;
    }

    sd::DataType Environment::defaultFloatDataType() {
        return _dataType.load();
    }

    std::vector<Pair>& Environment::capabilities() {
        return _capabilities;
    }

    void Environment::setDefaultFloatDataType(sd::DataType dtype) {
        if (dtype != sd::DataType::FLOAT32 && dtype != sd::DataType::DOUBLE && dtype != sd::DataType::FLOAT8 && dtype != sd::DataType::HALF)
            throw std::runtime_error("Default Float data type must be one of [FLOAT8, FLOAT16, FLOAT32, DOUBLE]");

        _dataType.store(dtype);
    }

    void Environment::setVerbose(bool reallyVerbose) {
        _verbose = reallyVerbose;
    }

    bool Environment::isDebug() {
        return _debug.load();
    }

    bool Environment::isProfiling() {
        return _profile.load();
    }

    bool Environment::isDetectingLeaks() {
        return _leaks.load();
    }

    void Environment::setLeaksDetector(bool reallyDetect) {
        _leaks.store(reallyDetect);
    }

    void Environment::setProfiling(bool reallyProfile) {
        _profile.store(reallyProfile);
    }

    bool Environment::isDebugAndVerbose() {
        return this->isDebug() && this->isVerbose();
    }

    void Environment::setDebug(bool reallyDebug) {
        _debug = reallyDebug;
    }

    int Environment::tadThreshold() {
        return _tadThreshold.load();
    }

    void Environment::setTadThreshold(int threshold) {
        _tadThreshold = threshold;
    }

    int Environment::elementwiseThreshold() {
        return _elementThreshold.load();
    }

    void Environment::setElementwiseThreshold(int threshold) {
        _elementThreshold = threshold;
    }

    int Environment::maxThreads() {
        return _maxThreads.load();
    }

    int Environment::maxMasterThreads() {
        return _maxMasterThreads.load();
    }

    void Environment::setMaxThreads(int max) {
        // FIXME: not possible at this moment, since maxThreads is limited by number of threads in pool. however we can allocate more threads if we want
        //_maxThreads.store(max);
    }

    void Environment::setMaxMasterThreads(int max) {
        if (max > maxThreads()) {
            max = maxThreads();
        }

        if (max < 1)
            return;

        _maxMasterThreads = max;
    }

    bool Environment::precisionBoostAllowed() {
        return _precBoost.load();
    }

    void Environment::allowPrecisionBoost(bool reallyAllow) {
        _precBoost.store(reallyAllow);
    }

    bool Environment::isCPU() {
#ifdef __CUDABLAS__
        return false;
#else
        return true;
#endif
    }

    int Environment::blasMajorVersion(){
        return _blasMajorVersion;
    }

    int Environment::blasMinorVersion(){
        return _blasMinorVersion;
    }

    int Environment::blasPatchVersion(){
        return _blasPatchVersion;
    }

    bool Environment::helpersAllowed() {
        return _allowHelpers.load();
    }

    void Environment::allowHelpers(bool reallyAllow) {
        _allowHelpers.store(reallyAllow);
    }

    void Environment::setGroupLimit(int group, Nd4jLong numBytes) {
        sd::memory::MemoryCounter::getInstance()->setGroupLimit((sd::memory::MemoryType) group, numBytes);
    }

    void Environment::setDeviceLimit(int deviceId, Nd4jLong numBytes) {
        sd::memory::MemoryCounter::getInstance()->setDeviceLimit(deviceId, numBytes);
    }

    Nd4jLong Environment::getGroupLimit(int group) {
        return sd::memory::MemoryCounter::getInstance()->groupLimit((sd::memory::MemoryType) group);
    }

    Nd4jLong Environment::getDeviceLimit(int deviceId) {
        return sd::memory::MemoryCounter::getInstance()->deviceLimit(deviceId);
    }

    Nd4jLong Environment::getGroupCounter(int group) {
        return sd::memory::MemoryCounter::getInstance()->allocatedGroup((sd::memory::MemoryType) group);
    }

    Nd4jLong Environment::getDeviceCounter(int deviceId) {
        return sd::memory::MemoryCounter::getInstance()->allocatedDevice(deviceId);
    }

    uint64_t Environment::maxPrimaryMemory() {
        return _maxTotalPrimaryMemory.load();
    }

    uint64_t Environment::maxSpecialMemory() {
        return _maxTotalSpecialMemory.load();
    }

    sd::Environment *sd::Environment::_instance = 0;

}
