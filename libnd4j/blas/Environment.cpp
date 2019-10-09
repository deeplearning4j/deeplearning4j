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
#include "Environment.h"
#include <helpers/StringUtils.h>

#ifdef _OPENMP

#include <omp.h>
#include <thread>
#include <helpers/logger.h>

#endif

#ifdef __CUDABLAS__

#include <cuda.h>
#include <cuda_runtime.h>
#include "BlasVersionHelper.h"
#endif

namespace nd4j {

    nd4j::Environment::Environment() {
        _tadThreshold.store(8);
        _elementThreshold.store(1024);
        _verbose.store(false);
        _debug.store(false);
        _profile.store(false);
        _precBoost.store(false);
        _leaks.store(false);
        _dataType.store(nd4j::DataType::FLOAT32);
        _maxThreads = std::thread::hardware_concurrency();

#ifndef ANDROID
        const char* omp_threads = std::getenv("OMP_NUM_THREADS");
        if (omp_threads != nullptr) {
            try {
                std::string omp(omp_threads);
                int val = std::stoi(omp);
                _maxThreads.store(val);
                nd4j_printf("Max threads: %i\n", val);
            } catch (std::invalid_argument &e) {
                // just do nothing
            } catch (std::out_of_range &e) {
                // still do nothing
            }
        }
#endif

#ifdef __CUDABLAS__
        BlasVersionHelper ver;
        _blasMajorVersion = ver._blasMajorVersion;
        _blasMinorVersion = ver._blasMinorVersion;
        _blasPatchVersion = ver._blasPatchVersion;
        printf("ND4J CUDA build version: %i.%i.%i\n", _blasMajorVersion, _blasMinorVersion, _blasPatchVersion);
        fflush(stdout);

        int devCnt = 0;
	    cudaGetDeviceCount(&devCnt);
	    auto devProperties = new cudaDeviceProp[devCnt];
	    for (int i = 0; i < devCnt; i++) {
		    cudaSetDevice(i);
		    cudaGetDeviceProperties(&devProperties[i], i);

		    //cudaDeviceSetLimit(cudaLimitStackSize, 4096);
		    Pair p(devProperties[i].major, devProperties[i].minor);
		    _capabilities.emplace_back(p);

		    printf("CUDA device %i: [%s]; cc: [%i.%i]; Total memory: [%lld];\n", i, devProperties[i].name, devProperties[i].major, devProperties[i].minor, (Nd4jLong) devProperties[i].totalGlobalMem);
	    }
	    fflush(stdout);

	    cudaSetDevice(0);
	    delete[] devProperties;
#else

#endif
    }

    nd4j::Environment::~Environment() {
        //
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

    nd4j::DataType Environment::defaultFloatDataType() {
        return _dataType.load();
    }

    std::vector<Pair>& Environment::capabilities() {
        return _capabilities;
    }

    void Environment::setDefaultFloatDataType(nd4j::DataType dtype) {
        if (dtype != nd4j::DataType::FLOAT32 && dtype != nd4j::DataType::DOUBLE && dtype != nd4j::DataType::FLOAT8 && dtype != nd4j::DataType::HALF)
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

    void Environment::setMaxThreads(int max) {
        _maxThreads.store(max);
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

    nd4j::Environment *nd4j::Environment::_instance = 0;

}
