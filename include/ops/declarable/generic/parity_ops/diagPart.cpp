//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 06.12.2017.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_diag_part)

#include <ops/declarable/CustomOperations.h>


namespace nd4j {
namespace ops  {
		
		CUSTOM_OP_IMPL(diag_part, 1, 1, false, 0, 0) {
			NDArray<T>* input  = INPUT_VARIABLE(0);
    		NDArray<T>* output = OUTPUT_VARIABLE(0);

    		const int inRank = input->rankOf();
    
    		// input validation
    		REQUIRE_TRUE(inRank == 2 ||  inRank == 4 || inRank == 6, 0, "DIAG_PART op: input array must have rank among following three possible values: 2, 4, 6, but got %i instead !", inRank);
    		for(int i = 0; i < inRank-1; ++i)
    			REQUIRE_TRUE(input->sizeAt(i) == input->sizeAt(i+1), 0, "DIAG_PART op: wrong shape of input array %s ! All dimensions must be equal !", ShapeUtils<T>::shapeAsString(input).c_str());

    		const int outLen = output->lengthOf();
    		const int inLen  = input->lengthOf();

    		int i(0), j(0);
    		while(j < outLen) {
    			(*output)(j) = (*input)(i);
    			i += outLen+1;
    			++j;
    		}
    
		    return ND4J_STATUS_OK;
		}
		DECLARE_SYN(DiagPart, diag_part);


		DECLARE_SHAPE_FN(diag_part) {

    		int* inputShapeInfo = inputShape->at(0);

    		const int inRank = inputShapeInfo[0];

    		int* outShapeInfo = nullptr;
	
			int outRank = inRank/2;
			if(outRank == 1)
				outRank += 1;
	
			ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(outRank), int);
	
			outShapeInfo[0] = outRank;
			for(int i = 1; i <= outRank; ++i)
				outShapeInfo[i] = inputShapeInfo[i];

			if(inRank/2 == 1)
				outShapeInfo[1] = 1;

			shape::updateStrides(outShapeInfo, shape::order(inputShapeInfo));

    		return SHAPELIST(outShapeInfo);
		}


}
}

#endif