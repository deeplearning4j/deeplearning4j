//
// CPU workspaces implementation
//
// @author raver119@gmail.com
//


#include <memory/Workspace.h>

namespace nd4j {
    namespace memory {

        Workspace::Workspace() : Workspace(0) {
            //
        }

        Workspace::Workspace(Nd4jIndex initialSize) {
            if (initialSize > 0) {
                this->_ptrHost = malloc(initialSize);

                if (this->_ptrHost == nullptr)
                    throw "Workspace allocation failed";
            } else
                this->_ptrHost = nullptr;

            this->_initialSize = initialSize;
            this->_currentSize = initialSize;
        }

        Workspace::~Workspace() {
            if (this->_ptrHost != nullptr)
                free(this->_ptrHost);
        }

        template <typename T>
        T* Workspace::allocateBytes(Nd4jIndex numBytes) {
            void* result = nullptr;
            this->_mutexAllocation.lock();

            if (_offset + numBytes >= _currentSize) {
                this->_mutexAllocation.unlock();
                return nullptr;
            }

            _offset += numBytes;
            result = _ptrHost + _offset;

            this->_mutexAllocation.unlock();

            return (T*) result;
        }


        template <typename T>
        T* Workspace::allocateBytes(MemoryType type, Nd4jIndex numBytes) {
            if (type == DEVICE)
                throw "CPU backend doesn't have device memory";

            return this->allocateBytes<T>(numBytes);
        }

        template <typename T>
        T* Workspace::allocateElements(Nd4jIndex numElements) {
            return this->allocateBytes<T>(numElements * sizeof(T));
        }

        template <typename T>
        T* Workspace::allocateElements(MemoryType type, Nd4jIndex numElements) {
            return this->allocateBytes<T>(type, numElements * sizeof(T));
        }
    }
}
