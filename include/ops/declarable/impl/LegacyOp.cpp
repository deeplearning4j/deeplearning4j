//
// Created by raver119 on 16.10.2017.
//

#include "ops/declarable/LegacyOp.h"


namespace nd4j {
    namespace ops {

        template <typename T>
        LegacyOp<T>::LegacyOp(int numInputs) : DeclarableOp<T>::DeclarableOp(numInputs , 1, "LegacyOp", true) {
            //
        }

        template <typename T>
        LegacyOp<T>::LegacyOp(int numInputs, int opNum) : DeclarableOp<T>::DeclarableOp(numInputs , 1, "LegacyOp", true) {
            _opNum = opNum;
        }


        template class LegacyOp<float>;
        template class LegacyOp<float16>;
        template class LegacyOp<double>;
    }
}