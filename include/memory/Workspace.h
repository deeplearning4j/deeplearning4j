//
// This class implements Workspace functionality in c++
//
//
// @author raver119@gmail.com
//

#ifndef LIBND4J_WORKSPACE_H
#define LIBND4J_WORKSPACE_H

#include <mutex>
#include <pointercast.h>

namespace nd4j {
    namespace memory {

        enum MemoryType {
            HOST,
            DEVICE,
        };

        class Workspace {
        protected:
            void* _ptrHost;
            void* _ptrDevice;

            Nd4jIndex _offset;

            Nd4jIndex _initialSize;
            Nd4jIndex _currentSize;

            std::mutex _mutexAllocation;

        public:
            Workspace();
            Workspace(Nd4jIndex initialSize);
            ~Workspace();

            bool resizeSupported();

            template <typename T>
            T* allocateBytes(Nd4jIndex numBytes);

            template <typename T>
            T* allocateBytes(MemoryType type, Nd4jIndex numBytes);

            template <typename T>
            T* allocateElements(Nd4jIndex numElements);

            template <typename T>
            T* allocateElements(MemoryType type, Nd4jIndex numBytes);

            void reset();
        };
    }
}

#endif //LIBND4J_WORKSPACE_H
