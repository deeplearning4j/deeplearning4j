//
// Created by Yurii Shyrma on 12.01.2018
//

#ifndef LIBND4J_HHCOLPICQR_H
#define LIBND4J_HHCOLPICQR_H

#include <ops/declarable/helpers/helpers.h>
#include <ops/declarable/helpers/hhColPivQR.h>
#include "NDArray.h"

namespace nd4j {
namespace ops {
namespace helpers {


template<typename T>
class HHcolPivQR {

    public:        
    
        NDArray<T> _qr;                    
        NDArray<T> _coeffs;
        NDArray<T> _permut;
        int _diagSize;

        HHcolPivQR(const NDArray<T>& matrix);

        void evalData();
};



}
}
}


#endif //LIBND4J_HHCOLPICQR_H
