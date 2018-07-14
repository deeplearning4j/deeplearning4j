//
//
//

#include <op_boilerplate.h>
#include <pointercast.h>
#include <graph/RandomGenerator.h>
#include <chrono>
#include <array/DataTypeUtils.h>

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
        int RandomGenerator::relativeT<int>(Nd4jLong index) {
            auto x = this->xoroshiro(index);
            auto r = static_cast<int>(x % DataTypeUtils::max<uint32_t>());
            return r;
        }

        template <typename T>
        T RandomGenerator::relativeT(Nd4jLong index, T from, T to) {
            return from + (this->relativeT<T>(index) * (to - from));
        }

        template <typename T>
        T RandomGenerator::relativeT(Nd4jLong index) {
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


        template int RandomGenerator::relativeT(Nd4jLong, int, int);
        template float RandomGenerator::relativeT(Nd4jLong, float, float);
        template double RandomGenerator::relativeT(Nd4jLong, double, double);
    }
}