//
//  @author raver119@gmail.com
//

#include <ops/declarable/generic/helpers/BroadcastHelper.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        
//////////////////////////////////////////////////////////////////////////		
		OP_IMPL(add, 2, 1, true) {
            NDArray<T> *x = INPUT_VARIABLE(0);
            NDArray<T> *y = INPUT_VARIABLE(1);
            NDArray<T> *z = OUTPUT_VARIABLE(0);

            auto tZ = BroadcastHelper<T>::template broadcast_apply<simdOps::Add<T>>(x, y, z);
            if (tZ == nullptr)
                return ND4J_STATUS_KERNEL_FAILURE;
            else if (tZ != z) {
                OVERWRITE_RESULT(tZ);
            }
    
			return ND4J_STATUS_OK;
        }
    }
}