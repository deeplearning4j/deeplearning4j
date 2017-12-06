//
//  @author raver119@gmail.com
//

#include <NDArray.h>
#include <NDArrayFactory.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
    namespace ops {
        template <typename T>
        class BroadcastHelper {
        public: 
            template <typename OpName>
            static FORCEINLINE NDArray<T>* broadcast_apply(NDArray<T>* x, NDArray<T>* y, NDArray<T>* z, T *extraArgs = nullptr) {
                if (!x->isScalar() && !y->isScalar() && x->lengthOf() == y->lengthOf()) {
				    x->template applyPairwiseTransform<OpName>(y, z, nullptr);
                } else if (!x->isScalar() && y->isScalar()) {
                    x->template applyScalar<OpName>(*y, z);
                } else if (x->isScalar() && !y->isScalar()) {
                    auto v = y->getShapeAsVector();
                    auto tZ = NDArrayFactory<T>::valueOf(v, x->getScalar(0), y->ordering());
                    tZ->template applyPairwiseTransform<OpName>(y, extraArgs);
                    return tZ;
                } else if (x->isScalar() && y->isScalar()) { // x->isScalar() && y->isScalar()
				    z->putScalar(0, OpName::op(x->getScalar(0), y->getScalar(0)));
			    } else if (ShapeUtils<T>::areShapesBroadcastable(*x, *y)) {
                    auto tZ = x->template applyTrueBroadcast<OpName>(y, extraArgs);
                    return tZ;
                } else {
                    auto sx = ShapeUtils<T>::shapeAsString(*x);
                    auto sy = ShapeUtils<T>::shapeAsString(*y);
                    nd4j_printf("RealDiv: shapes should be equal, or broadcastable. But got %s vs %s instead\n", sx.c_str(), sy.c_str());
                    return nullptr;
                }

                return z;
            }
        };
    }
}