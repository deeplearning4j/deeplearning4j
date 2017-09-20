//
// CPU workspaces implementation
//
// @author raver119@gmail.com
//


#include <atomic>
#include <stdio.h>
#include <stdlib.h>
#include "../Workspace.h"
#include <helpers/logger.h>
#include <templatemath.h>


namespace nd4j {
    namespace memory {

        Workspace::Workspace(Nd4jIndex initialSize) {
            if (initialSize > 0) {
                this->_ptrHost = (char *) malloc(initialSize);

                if (this->_ptrHost == nullptr)
                    throw "Workspace allocation failed";

                this->_allocatedHost = true;
            } else
                this->_allocatedHost = false;

            this->_initialSize = initialSize;
            this->_currentSize = initialSize;
            this->_offset = 0;
            this->_cycleAllocations = 0;
            this->_spillsSize = 0;
        }

        void Workspace::init(Nd4jIndex bytes) {
            if (this->_currentSize < bytes) {
                if (this->_allocatedHost)
                    free((void *)this->_ptrHost);

                this->_ptrHost =(char *) malloc(bytes);
                this->_currentSize = bytes;
                this->_allocatedHost = true;
            }
        }

        void Workspace::freeSpills() {
            _spillsSize = 0;

            if (_spills.size() < 1)
                return;

            for (auto v:_spills)
                free(v);

            _spills.clear();
        }

        Workspace::~Workspace() {
            if (this->_allocatedHost)
                free((void *)this->_ptrHost);

            freeSpills();
        }


        Nd4jIndex Workspace::getCurrentSize() {
            return _currentSize;
        }

        Nd4jIndex Workspace::getCurrentOffset() {
            return _offset.load();
        }


        void* Workspace::allocateBytes(Nd4jIndex numBytes) {
            void* result = nullptr;
            this->_cycleAllocations += numBytes;
            this->_mutexAllocation.lock();

            if (_offset.load() + numBytes > _currentSize) {
                this->_mutexAllocation.unlock();

                void *p = malloc(numBytes);

                _mutexSpills.lock();
                _spills.push_back(p);
                _mutexSpills.unlock();

                _spillsSize += numBytes;

                return p;
            }

            _offset += numBytes;
            result = (void *)(_ptrHost + _offset.load());

            this->_mutexAllocation.unlock();

            return result;
        }

        void Workspace::scopeIn() {
            freeSpills();
            init(_cycleAllocations.load());
            _cycleAllocations = 0;
        }

        void Workspace::scopeOut() {
            _offset = 0;
        }

        Nd4jIndex Workspace::getSpilledSize() {
            return _spillsSize.load();
        }

        void* Workspace::allocateBytes(nd4j::memory::MemoryType type, Nd4jIndex numBytes) {
            if (type == DEVICE)
                throw "CPU backend doesn't have device memory";

            return this->allocateBytes(numBytes);
        }

        Workspace* Workspace::clone() {
            // for clone we take whatever is higher: current allocated size, or allocated size of current loop
            Workspace* res = new Workspace(nd4j::math::nd4j_max<Nd4jIndex >(this->getCurrentSize(), this->_cycleAllocations.load()));
            return res;
        }
    }
}

