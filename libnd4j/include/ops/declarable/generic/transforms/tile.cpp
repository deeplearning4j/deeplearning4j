//
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_tile)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops {

CUSTOM_OP_IMPL(tile, 1, 1, false, 0, -2) {
    
    NDArray<T>* input  = INPUT_VARIABLE(0);
    NDArray<T>* output = OUTPUT_VARIABLE(0);    
    
    const int inRank = input->rankOf();
    std::vector<Nd4jLong> reps;

    if (block.getIArguments()->size() == inRank) {

        reps = ArrayUtils::toLongVector(*(block.getIArguments()));        
    } 
    else if (block.width() > 1)  {
        
        NDArray<T>* reps_vector = INPUT_VARIABLE(1);
        REQUIRE_TRUE(reps_vector->lengthOf() == inRank, 0, "TILE op: repeats vector length should be equal to input rank, but got %i and %i correspondingly !", reps_vector->lengthOf(), inRank);

        reps = reps_vector->template asVectorT<Nd4jLong>();
    }
    else {
        REQUIRE_TRUE(false, 0, "TILE op: this op requires repeats vector, either as IArgs or second array with length equal to rank of input array to be tiled !");
    }
            
    input->tile(reps, *output);

    return Status::OK();
}


DECLARE_SHAPE_FN(tile) {
    
    Nd4jLong* inShape = inputShape->at(0);
    const int inRank = inShape[0];
    std::vector<Nd4jLong> reps;

    if (block.getIArguments()->size() == inRank) {

        reps = ArrayUtils::toLongVector(*(block.getIArguments()));        
    } 
    else if (block.width() > 1)  {
        
        NDArray<T>* reps_vector = INPUT_VARIABLE(1);
        REQUIRE_TRUE(reps_vector->lengthOf() == inRank, 0, "TILE op: repeats vector length should be equal to input rank, but got %i and %i correspondingly !", reps_vector->lengthOf(), inRank);
        reps = reps_vector->template asVectorT<Nd4jLong>();
    }
    else {
        REQUIRE_TRUE(false, 0, "TILE op: this op requires repeats vector, either as IArgs or second array with length equal to rank of input array to be tiled !");
    }

    Nd4jLong* newShape;
    ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inRank), Nd4jLong);
    
    std::vector<Nd4jLong> shape(inRank);
    for (int e = 0; e < shape::rank(inShape); e++)
        shape[e] = shape::sizeAt(inShape, e) * reps[e];

    if (shape::order(inShape) == 'c')
        shape::shapeBuffer(shape.size(), shape.data(), newShape);
    else
        shape::shapeBufferFortran(shape.size(), shape.data(), newShape);
       
    return SHAPELIST(newShape);
}

}
}

#endif