//
// Created by raver119 on 11/06/18.
//

#include <graph/ResultWrapper.h>
#include <stdexcept>


namespace nd4j {
    namespace graph {
        ResultWrapper::ResultWrapper(Nd4jLong size, Nd4jPointer ptr) {
            if (size <= 0)
                throw std::runtime_error("FlatResult size should be > 0");

            _size = size;
            _pointer = ptr;
        }

        ResultWrapper::~ResultWrapper() {
            if (_pointer != nullptr && _size > 0) {
                auto ptr = reinterpret_cast<char *>(_pointer);
                delete[] ptr;
            }
        }


        Nd4jLong ResultWrapper::size() {
            return _size;
        }

        Nd4jPointer ResultWrapper::pointer() {
            return _pointer;
        }
    }
}