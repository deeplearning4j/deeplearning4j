//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_cumsum)

#include <ops/declarable/helpers/prefix.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops  {

CONFIGURABLE_OP_IMPL(cumsum, 1, 1, true, 0, 2) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    const bool exclusive = INT_ARG(0) == 1;
    const bool reverse = INT_ARG(1) == 1;

    if (block.getIArguments()->size() == 2 && block.width() == 1) {
        // all at once case
        nd4j::ops::helpers::_prefix<T, simdOps::Add<T>>(input->buffer(), input->shapeInfo(), output->buffer(), output->shapeInfo(), exclusive, reverse);
    } 
    else {
        std::vector<int> dims(block.numI() - 2);

        if (block.width() == 1) {

            for (int e = 0; e < block.numI() - 2; e++)
                dims[e] = INT_ARG(e + 2);
        } 
        else {
            auto ax = INPUT_VARIABLE(1);
            dims = ax->template asVectorT<int>();
        }

        for (int e = 0; e < dims.size(); e++)
            if (dims[e] < 0)
                dims[e] += input->rankOf();
        
        nd4j::ops::helpers::_prefix<T, simdOps::Add<T>>(input, output, dims, exclusive, reverse);
    }
    
    return ND4J_STATUS_OK;
}

CONFIGURABLE_OP_IMPL(cumsum_bp, 2, 1, true, 0, 2) {
//    auto input = INPUT_VARIABLE(0);
    auto gradOut = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);
//    output->assign(gradOut);
    const bool exclusive = INT_ARG(0) == 1;
    const bool reverse = INT_ARG(1) == 1;

    std::vector<int> dims;

    if (int newSize = (block.numI() - 2)) {
        dims.resize(newSize);

        for (int e = 0; e < newSize; e++)
            dims[e] = INT_ARG(e + 2);
    }
    if (!exclusive && !reverse) {
        if (dims.size())
            nd4j::ops::helpers::_prefix<T, simdOps::Add<T>>(gradOut, output, dims, false, true);
        else
            nd4j::ops::helpers::_prefix<T, simdOps::Add<T>>(gradOut->buffer(), gradOut->shapeInfo(), output->buffer(), output->shapeInfo(), false, true);

    }
    else if (!exclusive && reverse){
        if (dims.size())
            nd4j::ops::helpers::_prefix<T, simdOps::Add<T>>(gradOut, output, dims, false, false);
        else
            nd4j::ops::helpers::_prefix<T, simdOps::Add<T>>(gradOut->buffer(), gradOut->shapeInfo(), output->buffer(), output->shapeInfo(), false, false);
    }
    else if (exclusive && !reverse) {
        if (dims.size())
            nd4j::ops::helpers::_prefix<T, simdOps::Add<T>>(gradOut, output, dims, true, true);
        else
            nd4j::ops::helpers::_prefix<T, simdOps::Add<T>>(gradOut->buffer(), gradOut->shapeInfo(), output->buffer(), output->shapeInfo(), true, true);
    }
    else {
        if (dims.size())
            nd4j::ops::helpers::_prefix<T, simdOps::Add<T>>(gradOut, output, dims, true, false);
        else
            nd4j::ops::helpers::_prefix<T, simdOps::Add<T>>(gradOut->buffer(), gradOut->shapeInfo(), output->buffer(), output->shapeInfo(), true, false);
    }
        
    return ND4J_STATUS_OK;
}
}
}

#endif