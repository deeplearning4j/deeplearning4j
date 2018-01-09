//
// Created by raver119 on 09.01.18.
//

#ifndef LIBND4J_SHAPEBUILDER_H
#define LIBND4J_SHAPEBUILDER_H

#include <op_boilerplate.h>

namespace nd4j {
    class ShapeBuilder {
    public:
        static FORCEINLINE void shapeScalar(int *buffer) {
            buffer[0] = 0;
            buffer[1] = 0;
            buffer[2] = 1;
            buffer[3] = 99;
        }

        static FORCEINLINE void shapeVector(int length, int *buffer) {
            buffer[0] = 1;
            buffer[1] = length;
            buffer[2] = 1;
            buffer[3] = 0;
            buffer[4] = 1;
            buffer[5] = 99;
        }
    };
}



#endif //LIBND4J_SHAPEBUILDER_H
