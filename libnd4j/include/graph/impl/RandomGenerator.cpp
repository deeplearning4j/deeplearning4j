//
//
//

#include <op_boilerplate.h>
#include <pointercast.h>
#include <graph/RandomGenerator.h>
#include <chrono>
#include <array/DataTypeUtils.h>
#include <helpers/logger.h>

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
            _nodeState._long = nodeSeed;
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
            // This is default implementation for floating point types
            auto i = static_cast<float>(this->relativeT<int>(index));
            auto r = i / static_cast<float>(DataTypeUtils::max<int>());
            return static_cast<T>(r);
        }



        //////
        static FORCEINLINE uint32_t rotl(const uint32_t x, int k) {
	        return (x << k) | (x >> (32 - k));
        }

        static FORCEINLINE uint64_t rotl(const uint64_t x, int k) {
            return (x << k) | (x >> (64 - k));
        }

        uint32_t RandomGenerator::xoroshiro(Nd4jLong index) {
            u64 v;
            // TODO: improve this
            v._long = _rootState._long ^ _nodeState._long & index;

            return rotl(v._du32._v0 * 0x9E3779BB, 5) * 5;
        }

        void RandomGenerator::rewindH() {
            //
        }


        template int RandomGenerator::relativeT(Nd4jLong, int, int);
        template float16 RandomGenerator::relativeT(Nd4jLong, float16, float16);
        template float RandomGenerator::relativeT(Nd4jLong, float, float);
        template double RandomGenerator::relativeT(Nd4jLong, double, double);
    }
}