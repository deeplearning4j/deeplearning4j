//
// Created by raver119 on 06.10.2017.
//

#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include "Environment.h"

namespace nd4j {

    nd4j::Environment::Environment() {
        _tadThreshold.store(1);
        _elementThreshold.store(32);
        _verbose.store(false);
        _debug.store(false);

        const char* omp_threads = std::getenv("OMP_NUM_THREADS");
        if (omp_threads != nullptr) {
            try {
                std::string omp(omp_threads);
                int val = std::stoi(omp);
                _maxThreads.store(val);
            } catch (std::invalid_argument &e) {
                // just do nothing
            } catch (std::out_of_range &e) {
                // still do nothing
            }
        }
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

    void Environment::setVerbose(bool reallyVerbose) {
        _verbose = reallyVerbose;
    }

    bool Environment::isDebug() {
        return _debug.load();
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

    nd4j::Environment *nd4j::Environment::_instance = 0;

}
