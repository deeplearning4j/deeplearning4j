//
// CPU workspaces implementation
//
// @author raver119@gmail.com
//


#include <atomic>
#include <stdio.h>
#include <stdlib.h>
#include "../Workspace.h"


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
            this->_offset = 0;
        }

        Workspace::~Workspace() {
            if (this->_ptrHost != nullptr)
                free(this->_ptrHost);

            for (auto v:_spills)
                free(v);
        }


        Nd4jIndex Workspace::getCurrentSize() {
            return _currentSize;
        }

        Nd4jIndex Workspace::getCurrentOffset() {
            return _offset.load();
        }


        void* Workspace::allocateBytes(Nd4jIndex numBytes) {
            void* result = nullptr;
            this->_mutexAllocation.lock();

            if (_offset.load() + numBytes >= _currentSize) {
                this->_mutexAllocation.unlock();

                void *p = malloc(numBytes);

                _mutexSpills.lock();
                _spills.push_back(p);
                _mutexSpills.unlock();

                _spillsSize += numBytes;

                return p;
            }

            _offset += numBytes;
            result = _ptrHost + _offset.load();

            this->_mutexAllocation.unlock();

            return result;
        }



        void* Workspace::allocateBytes(nd4j::memory::MemoryType type, Nd4jIndex numBytes) {
            if (type == DEVICE)
                throw "CPU backend doesn't have device memory";

            return this->allocateBytes(numBytes);
        }
    }
}

