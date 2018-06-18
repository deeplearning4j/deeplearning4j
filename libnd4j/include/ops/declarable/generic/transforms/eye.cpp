//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 22.01.2018
//
#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_eye)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/transforms.h>

namespace nd4j {
namespace ops {

CUSTOM_OP_IMPL(eye, 0, 1, false, 0, 2) {

    helpers::eye(*OUTPUT_VARIABLE(0));

    return Status::OK();
}


DECLARE_SHAPE_FN(eye) {

    auto params = *block.getIArguments();
    const int size = params.size();

    Nd4jLong* outShapeInfo(nullptr);

    switch(size) {
        
        case 2:
            ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(2), Nd4jLong);
            outShapeInfo[0] = 2;
            outShapeInfo[1] = params[1];
            outShapeInfo[2] = params[1];
            break;

        case 3:
            ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(2), Nd4jLong);
            outShapeInfo[0] = 2;
            outShapeInfo[1] = params[1];
            outShapeInfo[2] = params[2];
            break;

        default:
            int rank = size-1;
            ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);
            outShapeInfo[0] = rank;
            outShapeInfo[rank-1] = params[1];
            outShapeInfo[rank] = params[2];
            for(int i = 1; i < rank-1; ++i)
                outShapeInfo[i] = params[i+2];
            break;
    }
        
    shape::updateStrides(outShapeInfo, (char)(params[0]));
        
    return SHAPELIST(outShapeInfo);
}


}
}

#endif