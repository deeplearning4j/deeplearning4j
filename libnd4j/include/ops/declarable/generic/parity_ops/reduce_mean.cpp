//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 01.06.2018
//


#include <ops/declarable/CustomOperations.h>


namespace nd4j    {
namespace ops     {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(reduce_mean, 1, 1, false, 0, -2) {

    return Status::OK();
}

DECLARE_SHAPE_FN(lstm) {    

    return SHAPELIST(nullptr);
}




}
}
