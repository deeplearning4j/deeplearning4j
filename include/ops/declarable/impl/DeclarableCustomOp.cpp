//
// Created by raver119 on 07.10.2017.
//

#include <ops/declarable/DeclarableCustomOp.h>
#include <ops/declarable/DeclarableOp.h>

namespace nd4j {
    namespace ops {
        template<typename T>
        DeclarableCustomOp<T>::DeclarableCustomOp(int numInputs, int numOutputs, const char *opName, bool allowsInplace, int tArgs, int iArgs) : nd4j::ops::DeclarableOp<T>(numInputs, numOutputs, opName, allowsInplace, tArgs, iArgs) {
            //
        }

        template<typename T>
        DeclarableCustomOp<T>::~DeclarableCustomOp()  {
            //
        }

        template class DeclarableCustomOp<float>;
        template class DeclarableCustomOp<float16>;
        template class DeclarableCustomOp<double>;
    }
}