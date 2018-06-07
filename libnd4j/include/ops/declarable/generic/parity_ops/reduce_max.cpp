//
// Created by george@skymind.io on 6/1/2018.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops {
#if NOT_EXCLUDED(OP_reduce_max)

    CUSTOM_OP_IMPL(reduce_max, 1, 1, false, 0, 0) {
        NDArray<T>* input = INPUT_VARIABLE(0);
        NDArray<T>* output = OUTPUT_VARIABLE(0);
        std::vector<int> axes = *block.getIArguments();

        for(const auto& item : axes)
            REQUIRE_TRUE(item > -input->shapeInfo()[0] || item <input->shapeInfo()[0], 0, "REDUCE_MEAN OP: the input dimension to reduce along must be in range (-%i, %i), but got %i instead !" , input->rankOf(), input->rankOf(), item);

        const bool keepDims = block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
        input->template reduceAlongDimension<simdOps::Max<T>>(output, axes, keepDims);

        return ND4J_STATUS_OK;
    }

    DECLARE_SHAPE_FN(reduce_max) {    

        const bool keepDims = block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
    
        std::vector<int> dimensions = *block.getIArguments();
        Nd4jLong* outShapeInfo = ShapeUtils<T>::evalReduceShapeInfo(shape::order(inputShape->at(0)), dimensions, inputShape->at(0), keepDims);

        return SHAPELIST(outShapeInfo);
    }
#endif 
#if NOT_EXCLUDED(OP_reduce_max_bp)

    DECLARE_SHAPE_FN(reduce_max_bp) {    

        const bool keepDims = block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
    
        Nd4jLong* outShapeInfo;// = ShapeUtils<T>::evalReduceShapeInfo(shape::order(inputShape->at(0)), dimensions, inputShape->at(0), keepDims);
        COPY_SHAPE(inputShape->at(0), outShapeInfo);

        return SHAPELIST(outShapeInfo);
    }

    CUSTOM_OP_IMPL(reduce_max_bp, 2, 1, false, 0, 0) {

            auto input = INPUT_VARIABLE(0);
            auto epsilon = INPUT_VARIABLE(1);
            auto output = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(output->isSameShape(epsilon), 0, "reduce_max_bp: The second param shape should be the same as result shape.");
            output->assign(epsilon);
            const bool keepDims = block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
            T keepDimsT = (keepDims?T(1.f):T(0.f));
            
            // at first step we build fwd activation
            nd4j::ops::reduce_max<T> op;
            std::vector<Nd4jLong> axes;

            if (block.numI() > 0) {
                for (int e = 0; e < block.numI(); e++)
                    axes.emplace_back(INT_ARG(e));// = *block.getIArguments();
            }
            std::vector<T> tVec(1);
            tVec[0] = (keepDims?T(1.0):T(0.0));
            std::vector<NDArray<T>*> inputVec({input});
            std::unique_ptr<ResultSet<T>> tmpResult(op.execute(inputVec, tVec, axes, false)); 
            if (tmpResult->status() != ND4J_STATUS_OK)
                return tmpResult->status();

            NDArray<T>* tempMax = tmpResult->at(0);
            REQUIRE_TRUE(tempMax->isSameShape(epsilon), 0, "reduce_max_bp: The second param shape should be the same with reduce_max output.");
            if (tempMax->isScalar()) {
                for (Nd4jLong e = 0; e < input->lengthOf(); e++) {
                    if (nd4j::math::nd4j_abs((*tempMax)(0) - (*input)(e)) < T(1.E-5f)) { // if input value equals to max
                         (*output)(e) = (*input)(e);
                    }
                }
            }
            else {
                for (Nd4jLong e = 0; e < input->lengthOf(); e++) {
                    for (Nd4jLong j = 0; e < tempMax->lengthOf(); j++) {
                        if (nd4j::math::nd4j_abs((*tempMax)(j) - (*input)(e)) < T(1.E-5f))  // if input value equals to max
                            (*output)(e) = (*input)(e);
                    }
                }
            }

            return ND4J_STATUS_OK;
    }
#endif

}
}
