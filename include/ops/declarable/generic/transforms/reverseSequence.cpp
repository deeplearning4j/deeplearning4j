//
//  Created by Yurii Shyrma on 25.01.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_reverse_sequence)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/reverseArray.h>

namespace nd4j {
namespace ops {

CUSTOM_OP_IMPL(reverse_sequence, 2, 1, false, 0, 2) {
        
    NDArray<T>* input      = INPUT_VARIABLE(0);
    NDArray<T>* seqLengths = INPUT_VARIABLE(1);
    NDArray<T>* output     = OUTPUT_VARIABLE(0);

    int seqDim = INT_ARG(0);
    int batchDim = block.numI() > 1 ? INT_ARG(1) : 0;

    REQUIRE_TRUE(input->rankOf() > 1, 0, "REVERSE_SEQUENSE custom operation: input array must have rank > 1 !");
    REQUIRE_TRUE(seqDim != batchDim, 0, "REVERSE_SEQUENSE custom operation: input integer parameters seqDim and batchDim must be different !");
    REQUIRE_TRUE(batchDim < input->rankOf(), 0, "REVERSE_SEQUENSE custom operation: input integer parameter batchDim must have value smaller than input array rank, but got %i instead !", batchDim);
    REQUIRE_TRUE(seqDim < input->rankOf(), 0, "REVERSE_SEQUENSE custom operation: input integer parameter seqDim must have value smaller than input array rank, but got %i instead !", seqDim);
    REQUIRE_TRUE(seqLengths->rankOf() == 1, 0, "REVERSE_SEQUENSE custom operation: input array seqLengths must be 1D vector, that is must have rank == 1, but got %i instead !", seqLengths->rankOf());
    REQUIRE_TRUE(seqLengths->lengthOf() == input->sizeAt(batchDim), 0, "REVERSE_SEQUENSE custom operation: the length of array seqLengths must be equal to the value of batchDim dimension of input array, but got length = %i instead !", seqLengths->lengthOf());

    T maxElem = seqLengths->template reduceNumber<simdOps::Max<T>>();
    REQUIRE_TRUE(maxElem <= (T)input->sizeAt(seqDim), 0, "REVERSE_SEQUENSE custom operation: max element in seqLengths array must be not greater than value of seqDim dimension of input array !");
    
    int posOfNonUnityDim = -1;
    if(input->isVector() || shape::isLikeVector(input->getShapeInfo(), posOfNonUnityDim)) {

        if((seqDim == 0 && input->sizeAt(0) == 1) || (batchDim == posOfNonUnityDim))
            output->assign(input);
        else 
            helpers::reverseArray<T>(input->getBuffer(), input->getShapeInfo(), output->getBuffer(), output->getShapeInfo(), (int)(*seqLengths)(0));
    }
    else {
            
        if(seqDim > batchDim)
            --seqDim;

        std::vector<int> dimensions = ShapeUtils<T>::evalDimsToExclude(input->rankOf(), {batchDim});       

        ResultSet<T>* inSubArrsSet  = NDArrayFactory<T>::allTensorsAlongDimension(input, dimensions);
        ResultSet<T>* outSubArrsSet = NDArrayFactory<T>::allTensorsAlongDimension(output, dimensions);

        for(int i = 0; i < inSubArrsSet->size(); ++i) {

            int numOfElemsToReverse = (*seqLengths)(i);
        
            if(numOfElemsToReverse == 0 || numOfElemsToReverse == 1) {
                outSubArrsSet->at(i)->assign(inSubArrsSet->at(i));
            }
            else {
                ResultSet<T>* inInnerSet  = NDArrayFactory<T>::allTensorsAlongDimension(inSubArrsSet->at(i), {seqDim});
                ResultSet<T>* outInnerSet = NDArrayFactory<T>::allTensorsAlongDimension(outSubArrsSet->at(i), {seqDim});
                for(int j = 0; j < inInnerSet->size(); ++j)
                    helpers::reverseArray<T>(inInnerSet->at(j)->getBuffer(), inInnerSet->at(j)->getShapeInfo(), outInnerSet->at(j)->getBuffer(), outInnerSet->at(j)->getShapeInfo(), numOfElemsToReverse);
            
                delete inInnerSet;
                delete outInnerSet;
            }
        }
        delete inSubArrsSet;
        delete outSubArrsSet;
    }

    return ND4J_STATUS_OK;
}


DECLARE_SHAPE_FN(reverse_sequence) {

    int* inShapeInfo = inputShape->at(0);
    
    int* outShapeInfo = nullptr;
    COPY_SHAPE(inShapeInfo, outShapeInfo);
        
    return SHAPELIST(outShapeInfo);
}


}
}

#endif