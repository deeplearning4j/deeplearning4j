//
// This class implements Workspace functionality in c++
//
//
// @author raver119@gmail.com
//

#ifndef LIBND4J_WORKSPACE_H
#define LIBND4J_WORKSPACE_H

#include <atomic>
#include <vector>
#include <mutex>
#include <dll.h>
#include <pointercast.h>
#include <types/float16.h>

namespace nd4j {
    namespace memory {

//        void ping();

        enum MemoryType {
            HOST,
            DEVICE,
        };

        class ND4J_EXPORT Workspace {
        protected:
            char* _ptrHost;
            char* _ptrDevice;

            bool _allocatedHost;
            bool _allocatedDevice;

            std::atomic<Nd4jIndex> _offset;

            Nd4jIndex _initialSize;
            Nd4jIndex _currentSize;

            std::mutex _mutexAllocation;
            std::mutex _mutexSpills;

            std::vector<void*> _spills;

            std::atomic<Nd4jIndex> _spillsSize;
            std::atomic<Nd4jIndex> _cycleAllocations;

            void init(Nd4jIndex bytes);
            void freeSpills();
        public:
            Workspace(Nd4jIndex initialSize = 0);
            ~Workspace();

            Nd4jIndex getAllocatedSize();
            Nd4jIndex getCurrentSize();
            Nd4jIndex getCurrentOffset();
            Nd4jIndex getSpilledSize();
            Nd4jIndex getUsedSize();

            void expandBy(Nd4jIndex numBytes);
            void expandTo(Nd4jIndex numBytes);

//            bool resizeSupported();

            void* allocateBytes(Nd4jIndex numBytes);
            void* allocateBytes(MemoryType type, Nd4jIndex numBytes);

            void scopeIn();
            void scopeOut();

            /*
             * This method creates NEW workspace of the same memory size and returns pointer to it
             */
            Workspace* clone();
        };
    }
}

#endif //LIBND4J_WORKSPACE_H
