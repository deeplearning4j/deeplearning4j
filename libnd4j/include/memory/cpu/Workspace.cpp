//
// CPU workspaces implementation
//
// @author raver119@gmail.com
//


#include <op_boilerplate.h>
#include <atomic>
#include <stdio.h>
#include <stdlib.h>
#include "../Workspace.h"
#include <helpers/logger.h>
#include <templatemath.h>
#include <cstring>


namespace nd4j {
    namespace memory {
        Workspace::Workspace(ExternalWorkspace *external) {
            if (external->sizeHost() > 0) {
                _ptrHost = (char *) external->pointerHost();
                _ptrDevice = (char *) external->pointerDevice();

                _initialSize = external->sizeHost();
                _currentSize = external->sizeHost();
                _offset = 0L;
                this->_cycleAllocations = 0;
                this->_spillsSize = 0;

                _externalized = true;
            }
        };

        Workspace::Workspace(Nd4jLong initialSize) {
            if (initialSize > 0) {
                this->_ptrHost = (char *) malloc(initialSize);

                CHECK_ALLOC(this->_ptrHost, "Failed to allocate new workspace");

                memset(this->_ptrHost, 0, initialSize);
                this->_allocatedHost = true;
            } else
                this->_allocatedHost = false;

            this->_initialSize = initialSize;
            this->_currentSize = initialSize;
            this->_offset = 0;
            this->_cycleAllocations = 0;
            this->_spillsSize = 0;
        }

        void Workspace::init(Nd4jLong bytes) {
            if (this->_currentSize < bytes) {
                if (this->_allocatedHost && !_externalized)
                    free((void *)this->_ptrHost);

                this->_ptrHost =(char *) malloc(bytes);

                CHECK_ALLOC(this->_ptrHost, "Failed to allocate new workspace");

                memset(this->_ptrHost, 0, bytes);
                this->_currentSize = bytes;
                this->_allocatedHost = true;
            }
        }

        void Workspace::expandBy(Nd4jLong numBytes) {
            this->init(_currentSize + numBytes);
        }

        void Workspace::expandTo(Nd4jLong numBytes) {
            this->init(numBytes);
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
            if (this->_allocatedHost && !_externalized)
                free((void *)this->_ptrHost);

            freeSpills();
        }

        Nd4jLong Workspace::getUsedSize() {
            return getCurrentOffset();
        }

        Nd4jLong Workspace::getCurrentSize() {
            return _currentSize;
        }

        Nd4jLong Workspace::getCurrentOffset() {
            return _offset.load();
        }


        void* Workspace::allocateBytes(Nd4jLong numBytes) {
            if (numBytes < 1) {
                nd4j_printf("Bad number of bytes requested for allocation: %i\n", numBytes);
                throw std::invalid_argument("Number of bytes for allocation should be positive");
            }

            //numBytes += 32;
            void* result = nullptr;
            this->_cycleAllocations += numBytes;
            this->_mutexAllocation.lock();

            if (_offset.load() + numBytes > _currentSize) {
                nd4j_debug("Allocating %lld bytes in spills\n", numBytes);
                this->_mutexAllocation.unlock();

                void *p = malloc(numBytes);

                CHECK_ALLOC(p, "Failed to allocate new workspace");

                _mutexSpills.lock();
                _spills.push_back(p);
                _mutexSpills.unlock();

                _spillsSize += numBytes;

                return p;
            }

            result = (void *)(_ptrHost + _offset.load());
            _offset += numBytes;
            //memset(result, 0, (int) numBytes);

            nd4j_debug("Allocating %lld bytes from workspace; Current PTR: %p; Current offset: %lld\n", numBytes, result, _offset.load());

            this->_mutexAllocation.unlock();

            return result;
        }

        Nd4jLong Workspace::getAllocatedSize() {
            return getCurrentSize() + getSpilledSize();
        }

        void Workspace::scopeIn() {
            freeSpills();
            init(_cycleAllocations.load());
            _cycleAllocations = 0;
        }

        void Workspace::scopeOut() {
            _offset = 0;
        }

        Nd4jLong Workspace::getSpilledSize() {
            return _spillsSize.load();
        }

        void* Workspace::allocateBytes(nd4j::memory::MemoryType type, Nd4jLong numBytes) {
            if (type == DEVICE)
                throw std::runtime_error("CPU backend doesn't have device memory");

            return this->allocateBytes(numBytes);
        }

        Workspace* Workspace::clone() {
            // for clone we take whatever is higher: current allocated size, or allocated size of current loop
            return new Workspace(nd4j::math::nd4j_max<Nd4jLong >(this->getCurrentSize(), this->_cycleAllocations.load()));
        }
    }
}

