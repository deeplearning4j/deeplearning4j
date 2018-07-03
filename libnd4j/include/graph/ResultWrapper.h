//
// Created by raver119 on 11/06/18.
//

#ifndef LIBND4J_RESULTWRAPPER_H
#define LIBND4J_RESULTWRAPPER_H

#include <op_boilerplate.h>
#include <pointercast.h>
#include <dll.h>

namespace nd4j {
    namespace graph {
        class ND4J_EXPORT ResultWrapper {
        private:
            Nd4jLong _size = 0L;
            Nd4jPointer _pointer = nullptr;

        public:
            ResultWrapper(Nd4jLong size, Nd4jPointer ptr);
            ~ResultWrapper();

            Nd4jLong size();

            Nd4jPointer pointer();
        };
    }
}


#endif //LIBND4J_RESULTWRAPPER_H
