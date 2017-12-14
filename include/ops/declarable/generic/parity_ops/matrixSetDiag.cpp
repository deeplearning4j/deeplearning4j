//
// Created by Yurii Shyrma on 07.12.2017.
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/matrixSetDiag.h>

namespace nd4j {
    namespace ops {
        CONFIGURABLE_OP_IMPL(matrix_set_diag, 2, 1, false, 0, 0) {
	        NDArray<T>* input    = INPUT_VARIABLE(0);
            NDArray<T>* diagonal = INPUT_VARIABLE(1);

	        NDArray<T>* output   = OUTPUT_VARIABLE(0);

            if(diagonal->rankOf() != input->rankOf()-1)
    	        throw "CONFIGURABLE_OP matrixSetDiag: rank of diagonal array must be smaller by one compared to rank of input array !";

            for(int i = 0;  i < diagonal->rankOf() - 1; ++i)        
    	        if(diagonal->sizeAt(i) != input->sizeAt(i))
    	            throw "CONFIGURABLE_OP matrixSetDiag: the shapes of diagonal and input arrays must be equal till last diagonal dimension but one !";

   	        if(diagonal->sizeAt(-1) != (int)nd4j::math::nd4j_min<Nd4jIndex>(input->sizeAt(-1), input->sizeAt(-2)))
    	        throw "CONFIGURABLE_OP matrixSetDiag: the shape of diagonal at last dimension must be equal to min(input_last_shape, input_last_but_one_shape) !";    

            *output = *input;

            const int lastDimSize = input->sizeAt(-1);
            const int last2DimSize = input->sizeAt(-1) * input->sizeAt(-2);
            const int lastSmallDim = diagonal->sizeAt(-1);
            const int batchSize = input->lengthOf()/last2DimSize;
    
// #pragma omp parallel for if(batchSize > Environment::getInstance()->elementwiseThreshold()) schedule(static) 
            for(int i = 0; i < batchSize; ++i )
                for(int j = 0; j < lastSmallDim; ++j) {
                    (*output)(i*last2DimSize + j*(lastDimSize + 1)) = (*diagonal)(i*lastSmallDim + j);            
                }
             
            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(MatrixSetDiag, matrix_set_diag);
    }
}