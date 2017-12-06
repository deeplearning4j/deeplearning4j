//
//  @author raver119@gmail.com
//

#include <ops/declarable/generic/helpers/BroadcastHelper.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        //////////////////////////////////////////////////////////////////////////		
		OP_IMPL(reversesubtract, 2, 1, true) {
            NDArray<T> *x = INPUT_VARIABLE(0);
            NDArray<T> *y = INPUT_VARIABLE(1);
            NDArray<T> *z = this->getZ(block);

            auto tZ = BroadcastHelper<T>::template broadcast_apply<simdOps::ReverseSubtract<T>>(x, y, z);
            if (tZ == nullptr)
                return ND4J_STATUS_KERNEL_FAILURE;
            else if (tZ != z) {
                OVERWRITE_RESULT(tZ);
            }

            /*
			if (!x->isScalar() && !y->isScalar() && x->lengthOf() == y->lengthOf()) {
				REQUIRE_OK(this->validateInputLengthMatch(block));
				x->template applyPairwiseTransform<simdOps::ReverseSubtract<T>>(y, z, nullptr);
            
            } else if (!x->isScalar() && y->isScalar()) {
               x->template applyScalar<simdOps::ReverseSubtract<T>>(*y, z);

            } else if (x->isScalar() && !y->isScalar()) {
                y->template applyScalar<simdOps::ReverseSubtract<T>>(*x, z);

            }						
			else if (x->isScalar() && y->isScalar()) { // x->isScalar() && y->isScalar()
				z->putScalar(0, y->getScalar(0) - x->getScalar(0));
			} else if (ShapeUtils<T>::areShapesBroadcastable(*x, *y)) {
                auto tZ = x->template applyTrueBroadcast<simdOps::ReverseSubtract<T>>(y);
                OVERWRITE_RESULT(tZ);
            } else {
                auto sx = ShapeUtils<T>::shapeAsString(*x);
                auto sy = ShapeUtils<T>::shapeAsString(*y);
                REQUIRE_TRUE(false, 0, "ReverseSubtract: shapes should be equal, or broadcastable. But got %s vs %s instead", sx.c_str(), sy.c_str());
            }
            */

			return ND4J_STATUS_OK;
        }
        DECLARE_SYN(RSub, reversesubtract);
    }
}