//
// Created by Yurii Shyrma on 27.01.2018
//

#include <helpers/ProviderRNG.h>
#include <NativeOps.h>


namespace nd4j {
    
ProviderRNG::ProviderRNG() {

    Nd4jIndex *buffer = new Nd4jIndex[100000];
    NativeOps nativeOps;    
    std::lock_guard<std::mutex> lock(_mutex);
    _rng = (nd4j::random::RandomBuffer *) nativeOps.initRandom(nullptr, 123, 100000, (Nd4jPointer) buffer);    
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