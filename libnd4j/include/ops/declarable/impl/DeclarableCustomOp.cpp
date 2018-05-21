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

        template class ND4J_EXPORT DeclarableCustomOp<float>;
        template class ND4J_EXPORT DeclarableCustomOp<float16>;
        template class ND4J_EXPORT DeclarableCustomOp<double>;
    }
}