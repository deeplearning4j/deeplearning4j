//
// Stronger 64-bit hash function helper, as described here: http://www.javamex.com/tutorials/collections/strong_hash_code_implementation.shtml
// @author raver119@gmail.com
//

#ifndef LIBND4J_HELPER_HASH_H
#define LIBND4J_HELPER_HASH_H

#include <string>
#include <dll.h>
#include <pointercast.h>
#include <mutex>

namespace nd4j {
    namespace ops {
        class ND4J_EXPORT HashHelper {
        private:
            static HashHelper* _INSTANCE;

            Nd4jLong _byteTable[256];
            const Nd4jLong HSTART = 0xBB40E64DA205B064L;
            const Nd4jLong HMULT = 7664345821815920749L;

            bool _isInit = false;
            std::mutex _locker;

        public:
            static HashHelper* getInstance();
            Nd4jLong getLongHash(std::string& str);
        };
    }
}

#endif //LIBND4J_HELPER_HASH_H
