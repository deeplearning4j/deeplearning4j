//
//
//

#include <pointercast.h>
#include <graph/RandomGenerator.h>
#include <chrono>

namespace nd4j {
    namespace graph {
        RandomGenerator::RandomGenerator(Nd4jLong rootSeed, Nd4jLong nodeSeed) {
            
            this->setStates(rootSeed, nodeSeed);
        }
        
        RandomGenerator::~RandomGenerator() {
            //
        }

        void RandomGenerator::setStates(Nd4jLong rootSeed, Nd4jLong nodeSeed) {
            // this seed is used graph-level state
            if (rootSeed == 0)
                rootSeed = currentMilliseconds();

            // graph-level state is just first seed
            _rootState._long = rootSeed;

            // used to build second, node state
        }


        Nd4jLong RandomGenerator::currentMilliseconds() {
            auto s = std::chrono::system_clock::now().time_since_epoch();
            auto v = std::chrono::duration_cast<std::chrono::milliseconds>(s).count();
            return v;
        }

        template <>
        int RandomGenerator::relativeT(Nd4jLong index, int from, int to) {
            return 0;
        }

        template <typename T>
        T RandomGenerator::relativeT(Nd4jLong index, T from, T to) {
            return static_cast<T>(0);
        }





        //////
        static FORCEINLINE uint32_t rotl(const uint32_t x, int k) {
	        return (x << k) | (x >> (32 - k));
        }

        uint32_t RandomGenerator::xoroshiro(Nd4jLong index) {
            u64 v;
            v._long = index;
        }

        void RandomGenerator::rewindH() {
            //
        }
    }
}