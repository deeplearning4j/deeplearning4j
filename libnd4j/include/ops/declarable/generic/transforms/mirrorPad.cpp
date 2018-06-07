//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 07.06.2018
//
#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_mirror_pad)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/transforms.h>

namespace nd4j {
namespace ops {


CUSTOM_OP_IMPL(mirror_pad, 2, 1, false, 0, 1) {

    NDArray<T>* input    = INPUT_VARIABLE(0);
    NDArray<T>* paddings = INPUT_VARIABLE(1);
    
    NDArray<T>* output  = OUTPUT_VARIABLE(0);

    const int mode = INT_ARG(0);    // 0 - REFLECT, else - SYMMETRIC
    const int includeBorder = mode ? 0 : 1;
    const int delta         = mode ? 1 : 0;

    REQUIRE_TRUE(paddings->rankOf() == 2, 0, "MIRROR_PAD OP: the rank of paddings array must be equal 2, but got %i instead !", paddings->rankOf());
    REQUIRE_TRUE(paddings->sizeAt(0) == input->rankOf(), 0, "MIRROR_PAD OP: zero dimension of paddings array must be equal input array rank, but got %i and %i correspondingly !", paddings->sizeAt(0), input->rankOf());
    for(int i = 0; i < input->rankOf(); ++i)
        REQUIRE_TRUE( ((*paddings)(i,0) <= (input->sizeAt(i) - includeBorder)) && ((*paddings)(i,1) <= (input->sizeAt(i) - includeBorder)), 0, "MIRROR_PAD OP: wrong content of paddings array, its elements must be no grater then corresponding dimensions of input array!");    
    
    const int rank = input->rankOf();
    std::vector<Nd4jLong> inIdx(rank), outIdx(rank);
    
    for(int i = 0; i < output->lengthOf(); ++i) {

        shape::ind2subC(rank, output->shapeOf(), i, outIdx.data());

        for(int j = 0; j < rank; ++j) {

            const int currInDim  = input->sizeAt(j);
            const int leftSide   = static_cast<int>((*paddings)(j, 0));
            const int wholeRange = leftSide + currInDim + static_cast<int>((*paddings)(j, 1));

            if(outIdx[j] < leftSide) 
                inIdx[j] = leftSide - outIdx[j] - delta;

            else if(outIdx[j] >= leftSide && outIdx[j] < leftSide + currInDim) 
                inIdx[j] = outIdx[j] - leftSide;

            else
                inIdx[j] = wholeRange - outIdx[j] - includeBorder;
        }
        
        Nd4jLong outOffset = shape::getOffset(0, output->shapeOf(), output->stridesOf(), outIdx.data(), rank);
        Nd4jLong inOffset  = shape::getOffset(0, input->shapeOf(),  input->stridesOf(),  inIdx.data(),  rank);
        output->getBuffer()[outOffset] = input->getBuffer()[inOffset];
    }

    return Status::OK();
}
 

DECLARE_SHAPE_FN(mirror_pad) {

    NDArray<T>* input    = INPUT_VARIABLE(0);
    NDArray<T>* paddings = INPUT_VARIABLE(1);

    const int rank = input->rankOf();
    const int includeBorder = static_cast<bool>(INT_ARG(0)) ? 0 : 1;

    REQUIRE_TRUE(paddings->rankOf() == 2, 0, "MIRROR_PAD OP: the rank of paddings array must be equal 2, but got %i instead !", paddings->rankOf());
    REQUIRE_TRUE(paddings->sizeAt(0) == rank, 0, "MIRROR_PAD OP: zero dimension of paddings array must be equal input array rank, but got %i and %i correspondingly !", paddings->sizeAt(0), rank);
    for(int i = 0; i < rank; ++i)
        REQUIRE_TRUE( ((*paddings)(i,0) <= (input->sizeAt(i) - includeBorder)) && ((*paddings)(i,1) <= (input->sizeAt(i) - includeBorder)), 0, "MIRROR_PAD OP: wrong content of paddings array, its elements must be no grater then corresponding dimensions of input array!");    
    
    Nd4jLong* outShapeInfo(nullptr);
    ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);

    outShapeInfo[0] = rank;
    for(int i = 0; i < rank; ++i)
        outShapeInfo[i+1] = input->sizeAt(i) + (*paddings)(i,0) + (*paddings)(i,1);

    shape::updateStrides(outShapeInfo, input->ordering());

    return SHAPELIST(outShapeInfo);
}


}
}

#endif