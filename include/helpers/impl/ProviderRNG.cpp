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