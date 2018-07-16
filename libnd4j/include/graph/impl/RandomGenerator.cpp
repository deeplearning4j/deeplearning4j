//
// @author raver119@protonmail.com
//
// relies on xoroshiro 64** and xoroshiro128 implementations

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
        uint64_t RandomGenerator::relativeT<uint64_t>(Nd4jLong index) {
            return this->xoroshiro64(index);
        }

        template <>
        uint32_t RandomGenerator::relativeT<uint32_t>(Nd4jLong index) {
            return this->xoroshiro32(index);
        }

        template <>
        int RandomGenerator::relativeT<int>(Nd4jLong index) {
            auto x = this->relativeT<uint32_t>(index);
            auto r = static_cast<int>(x % DataTypeUtils::max<int>());
            return r;
        }

        template <>
        Nd4jLong RandomGenerator::relativeT<Nd4jLong>(Nd4jLong index) {
            auto x = this->relativeT<uint64_t>(index);
            auto r = static_cast<Nd4jLong>(x % DataTypeUtils::max<Nd4jLong>());
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

        uint32_t RandomGenerator::xoroshiro32(Nd4jLong index) {
            u64 v;
            // TODO: improve this
            v._long = _rootState._long ^ _nodeState._long ^ index;

            return rotl(v._du32._v0 * 0x9E3779BB, 5) * 5;
        }

        uint64_t RandomGenerator::xoroshiro64(Nd4jLong index) {
            auto s0 = _rootState._ulong;
            auto s1 = _nodeState._ulong;

            // xor by idx
            _nodeState._long ^= index;

            // since we're not modifying state - do rotl step right here
            s1 ^= s0;
            s0 = rotl(s0, 55) ^ s1 ^ (s1 << 14);
            s1 = rotl(s1, 36);

            return s0 + s1;
        }

        void RandomGenerator::rewindH(Nd4jLong steps) {
            auto s0 = _nodeState._du32._v0;
            auto s1 = _nodeState._du32._v1;

            s1 ^= s0;
	        _nodeState._du32._v0 = rotl(s0, 26) ^ s1 ^ (s1 << 9); // a, b
	        _nodeState._du32._v1 = rotl(s1, 13); // c

            // TODO: improve this
            _nodeState._long ^= steps;
        }


        template int RandomGenerator::relativeT(Nd4jLong, int, int);
        template float16 RandomGenerator::relativeT(Nd4jLong, float16, float16);
        template float RandomGenerator::relativeT(Nd4jLong, float, float);
        template double RandomGenerator::relativeT(Nd4jLong, double, double);
        template Nd4jLong RandomGenerator::relativeT(Nd4jLong, Nd4jLong, Nd4jLong);
    }
}