//
//  @author raver119@gmail.com
//

#ifndef LIBND4J_BROADCAST_HELPER_H
#define LIBND4J_BROADCAST_HELPER_H

#include <NDArray.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
    namespace ops {
        template <typename T>
        class BroadcastHelper {
        public: 
            template <typename OpName>
            static FORCEINLINE NDArray<T>* broadcast_apply(NDArray<T>* x, NDArray<T>* y, NDArray<T>* z, T *extraArgs = nullptr) {
                if (!x->isScalar() && !y->isScalar() && x->isSameShape(y)) {
				    x->template applyPairwiseTransform<OpName>(y, z, nullptr);
                } else if (!x->isScalar() && y->isScalar()) {
                    x->template applyScalar<OpName>(*y, z);
                } else if (x->isScalar() && !y->isScalar()) {
                    if (z->isSameShape(y)) {
                        z->assign(x);
                        z->template applyPairwiseTransform<OpName>(y, extraArgs);
                        return z;
                    } else {
                        auto v = y->getShapeAsVector();
                        auto tZ = NDArray<T>::valueOf(v, x->getScalar(0), y->ordering());
                        tZ->template applyPairwiseTransform<OpName>(y, extraArgs);
                        return tZ;
                    }
                } else if (x->isScalar() && y->isScalar()) { // x->isScalar() && y->isScalar()
				    z->putScalar(0, OpName::op(x->getScalar(0), y->getScalar(0)));
			    } else if (ShapeUtils<T>::areShapesBroadcastable(*x, *y)) {
                    x->template applyTrueBroadcast<OpName>(y, z, true, extraArgs);
                    return z;
                } else {
                    auto sx = ShapeUtils<T>::shapeAsString(x);
                    auto sy = ShapeUtils<T>::shapeAsString(y);
                    nd4j_printf("RealDiv: shapes should be equal, or broadcastable. But got %s vs %s instead\n", sx.c_str(), sy.c_str());
                    return nullptr;
                }

                return z;
            }
        };
    }
}

#endif