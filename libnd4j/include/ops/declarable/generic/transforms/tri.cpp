//
// @author Yurii Shyrma, created on 31.03.2018
//

#include <ops/declarable/CustomOperations.h>


namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(tri, -2, 1, false, 0, 1) {
	
    NDArray<T>* output = OUTPUT_VARIABLE(0);

    const int diag = block.getIArguments()->size() > 2 ? INT_ARG(2) : 0;

    output->setValueInDiagMatrix(1., diag,   'l');          // fill with unities lower triangular block of matrix
    output->setValueInDiagMatrix(0., diag+1, 'u');          // fill with zeros upper triangular block of matrix

    return Status::OK();
}


DECLARE_SHAPE_FN(tri) {

	const int rows = INT_ARG(0);
    const int cols = block.getIArguments()->size() > 1 ? INT_ARG(1) : rows;
    const int rank = 2;

    Nd4jLong* outShapeInfo = nullptr;
	ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);

    outShapeInfo[0] = rank;
    outShapeInfo[1] = rows;
    outShapeInfo[2] = cols;

	shape::updateStrides(outShapeInfo, 'c');

    return SHAPELIST(outShapeInfo);    
}




}
}