//
// @author Yurii Shyrma, created on 31.03.2018
//

#include <ops/declarable/CustomOperations.h>


namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(triu, 1, 1, false, 0, 0) {
	
    NDArray<T>* input  = INPUT_VARIABLE(0);
    NDArray<T>* output = OUTPUT_VARIABLE(0);

    const int diag = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
    const int rank = input->rankOf();
    
    switch(rank) {

        case 1:
            for(int i = 0; i < output->sizeAt(0); ++i)
                (*output)({{i, i+1}, {}}).assign(input);
            output->setValueIn2DMatrix(0., diag-1, 'l');    
            break;

        case 2:
            output->assign(input);
            output->setValueIn2DMatrix(0., diag-1, 'l');    
            break;

        default:            
            ResultSet<T>* inTads  = NDArrayFactory<T>::allTensorsAlongDimension(input,  {rank-2, rank-1});
            ResultSet<T>* outTads = NDArrayFactory<T>::allTensorsAlongDimension(output, {rank-2, rank-1});            
            for(int i = 0; i < inTads->size(); ++i) {
                outTads->at(i)->assign(inTads->at(i));
                outTads->at(i)->setValueIn2DMatrix(0., diag-1, 'l');    
            }
            delete inTads;
            delete outTads;
    }

    return Status::OK();
}


DECLARE_SHAPE_FN(triu) {

	int* inShapeInfo = inputShape->at(0);

    int rank = (inShapeInfo[0] == 1) ? 2 : inShapeInfo[0];
    
    int* outShapeInfo = nullptr;
	ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), int);    
    memcpy(outShapeInfo, inShapeInfo, (1 + rank) * sizeof(int));                     // copy rank and dimensions values only

    if(inShapeInfo[0] == 1) {
        outShapeInfo[0] = rank; 
        outShapeInfo[1] = inShapeInfo[1];
        outShapeInfo[2] = inShapeInfo[1];
    }

	shape::updateStrides(outShapeInfo, shape::order(inShapeInfo));

    return SHAPELIST(outShapeInfo);    
}



//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(triu_bp, 2, 1, false, 0, 0) {
    
    NDArray<T>* input = INPUT_VARIABLE(0);
    NDArray<T>* dLdO = INPUT_VARIABLE(1);

    NDArray<T>* dLdI = OUTPUT_VARIABLE(0);              // dLoss/dI

    const int diag = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;

    nd4j::ops::triu<T> op;
    ResultSet<T>* results = op.execute({input}, {}, {diag});
    NDArray<T>* dOdI = results->at(0);                          // dO/dI
    
    for(int i = 0; i < dOdI->lengthOf(); ++i) {
        T* currElement = &(*dOdI)(i);
        if(*currElement != (T)0.)
            *currElement = 1.;
    }

    dLdI->assign((*dOdI) * (*dLdO));                          // chain rule: dLoss/dI = dO/dI * dLoss/dO 

    delete results;
    return Status::OK();
}


DECLARE_SHAPE_FN(triu_bp) {

    int* gradOShapeInfo = inputShape->at(1);
    int rank = gradOShapeInfo[0];

    int* outShapeInfo = nullptr;
    ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), int);    
    memcpy(outShapeInfo, gradOShapeInfo, (1 + rank) * sizeof(int));                     // copy rank and dimensions values only    

    shape::updateStrides(outShapeInfo, shape::order(inputShape->at(0)));

    return SHAPELIST(outShapeInfo);    
}


}
}