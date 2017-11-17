//
//  @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        //////////////////////////////////////////////////////////////////////////
		// here iArgs is int vector of repeats at the beginning and last element in iArgs is dimension
		CUSTOM_OP_IMPL(repeat, 1, 1, true, 0, -1) {			

			NDArray<T>* x   = INPUT_VARIABLE(0);
            NDArray<T>* ret = OUTPUT_VARIABLE(0);

			x->repeat(block.getIArguments()->back(), *ret);
			STORE_RESULT(*ret);

			return ND4J_STATUS_OK;				
        }
		
        DECLARE_SHAPE_FN(repeat) {                               
            
            NDArray<T>* x   = INPUT_VARIABLE(0);
            std::vector<int>* argumets = block.getIArguments();
            int argsSize = argumets->size();
            int dimension = (*argumets)[argsSize-1];
            std::vector<int> repeats = *argumets;
            repeats.pop_back();
            
            std::vector<int> outShape = ShapeUtils<T>::evalRepeatShape(dimension, repeats, *x);
            int rank = outShape.size();

            int* newShapeInfo = nullptr; 
            ALLOCATE(newShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), int); 
            newShapeInfo[0] = rank;
            copy(outShape.begin(), outShape.end(), newShapeInfo+1);
            shape::updateStrides(newShapeInfo, x->ordering());

            return new ShapeList(newShapeInfo);
        }
    }
}