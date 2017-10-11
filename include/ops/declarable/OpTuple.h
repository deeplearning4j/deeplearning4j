//
// Created by raver119 on 11.10.2017.
//

#ifndef LIBND4J_OPTUPLE_H
#define LIBND4J_OPTUPLE_H

#include <vector>
#include <initializer_list>
#include <NDArray.h>

namespace nd4j {
    namespace ops {
        class OpTuple {
        public:
            const char * _opName;
            std::initializer_list<nd4j::NDArray<float> *> _inputs;
            std::initializer_list<nd4j::NDArray<float> *> _outputs;
            std::initializer_list<float> _tArgs;
            std::initializer_list<int> _iArgs;

            OpTuple(const char *opName, std::initializer_list<nd4j::NDArray<float> *>&& inputs, std::initializer_list<float>&& tArgs, std::initializer_list<int>&& iArgs);
            ~OpTuple() = default;
        };
    }
}


#endif //LIBND4J_OPTUPLE_H
