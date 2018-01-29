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
