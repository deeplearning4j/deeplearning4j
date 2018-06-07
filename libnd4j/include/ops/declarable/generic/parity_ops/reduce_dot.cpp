
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops {
#if NOT_EXCLUDED(OP_reduce_dot_bp)

    DECLARE_SHAPE_FN(reduce_dot_bp) {    

        const bool keepDims = block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
    
        Nd4jLong* outShapeInfo;// = ShapeUtils<T>::evalReduceShapeInfo(shape::order(inputShape->at(0)), dimensions, inputShape->at(0), keepDims);
        COPY_SHAPE(inputShape->at(0), outShapeInfo);

        return SHAPELIST(outShapeInfo);
    }

    CUSTOM_OP_IMPL(reduce_dot_bp, 2, 1, false, 0, 0) {

            auto input = INPUT_VARIABLE(0);
            auto epsilon = INPUT_VARIABLE(1);
            auto output = OUTPUT_VARIABLE(0);

//            REQUIRE_TRUE(output->isSameShape(epsilon), 0, "reduce_dot_bp: The second param shape should be the same as result shape.");
//            output->assign(epsilon);
//            const bool keepDims = block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
//            T keepDimsT = (keepDims?T(1.f):T(0.f));
            // at first step we build fwd activation
/*
            nd4j::ops::reduce_dot<T> op;
            std::vector<Nd4jLong> axes;

            if (block.numI() > 0) {
                for (int e = 0; e < block.numI(); e++)
                    axes.emplace_back(INT_ARG(e));// = *block.getIArguments();
            }
            std::vector<T> tVec(1);
            tVec[0] = (keepDims?T(1.0):T(0.0));
            std::vector<NDArray<T>*> inputVec({input});
            auto tmpResult = op.execute(inputVec, tVec, axes, false); 
            if (tmpResult->status() != ND4J_STATUS_OK)
                return tmpResult->status();

            NDArray<T>* tempSum = tmpResult->at(0);
            tempSum->template applyPairwiseTransform<simdOps::Multiply<T>>(epsilon, output, nullptr);

            delete tmpResult;
*/
            return ND4J_STATUS_OK;
    }
#endif

}
}
