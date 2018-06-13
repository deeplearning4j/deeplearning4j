//
// Created by george@skymind.io on 6/4/2018.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops {
#if NOT_EXCLUDED(OP_reduce_norm2)

    CUSTOM_OP_IMPL(reduce_norm2, 1, 1, false, 0, 0) {
        NDArray<T>* input = INPUT_VARIABLE(0);
        NDArray<T>* output = OUTPUT_VARIABLE(0);
        std::vector<int> axes = *block.getIArguments();

        for(const auto& item : axes)
            REQUIRE_TRUE(item > -input->shapeInfo()[0] || item <input->shapeInfo()[0], 0, "REDUCE_MEAN OP: the input dimension to reduce along must be in range (-%i, %i), but got %i instead !" , input->rankOf(), input->rankOf(), item);

        const bool keepDims = block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
        input->template reduceAlongDimension<simdOps::Norm2<T>>(output, axes, keepDims);

        return ND4J_STATUS_OK;
    }

    DECLARE_SHAPE_FN(reduce_norm2) {    

        const bool keepDims = block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
    
        std::vector<int> dimensions = *block.getIArguments();
        Nd4jLong* outShapeInfo = ShapeUtils<T>::evalReduceShapeInfo(shape::order(inputShape->at(0)), dimensions, inputShape->at(0), keepDims, false, block.getWorkspace());

        return SHAPELIST(outShapeInfo);
    }
#endif 
#if NOT_EXCLUDED(OP_reduce_norm2_bp)

    DECLARE_SHAPE_FN(reduce_norm2_bp) {    

        const bool keepDims = block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
    
        Nd4jLong* outShapeInfo;// = ShapeUtils<T>::evalReduceShapeInfo(shape::order(inputShape->at(0)), dimensions, inputShape->at(0), keepDims, false, block.getWorkspace());
        COPY_SHAPE(inputShape->at(0), outShapeInfo);

        return SHAPELIST(outShapeInfo);
    }

    CUSTOM_OP_IMPL(reduce_norm2_bp, 2, 1, false, 0, 0) {

            auto input = INPUT_VARIABLE(0);
            auto epsilon = INPUT_VARIABLE(1);
            auto output = OUTPUT_VARIABLE(0);

            const bool keepDims = block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
            T keepDimsT = (keepDims?T(1.f):T(0.f));

            // at first step we build fwd activation
            nd4j::ops::reduce_norm2<T> op;
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

            NDArray<T>* tempNorm2 = tmpResult->at(0);


            if (tempNorm2->isScalar()) {
                auto norm2Backprop = LAMBDA_T(_x, epsilon, tempNorm2) {
                    return (*epsilon)(0) * _x / (*tempNorm2)(0);
                };
                input->applyLambda(norm2Backprop, output);
            }
            else {
                std::vector<int> dimensions; //(input->rankOf() - axes.size());
                for (Nd4jLong e = 0; e < input->rankOf(); e++) {
                    if (std::find(axes.begin(), axes.end(), e) == axes.end()) {
                        dimensions.emplace_back(e);
                    }
                }
                std::unique_ptr<ResultSet<T>> outList(NDArrayFactory<T>::allTensorsAlongDimension(output, dimensions));
                std::unique_ptr<ResultSet<T>> inList(NDArrayFactory<T>::allTensorsAlongDimension(input, dimensions));
                for (int e = 0; e < outList->size(); ++e) {
                    epsilon->template applyPairwiseTransform<simdOps::Multiply<T>>(inList->at(e), outList->at(e), nullptr);
                    outList->at(e)->template applyPairwiseTransform<simdOps::Divide<T>>(tempNorm2, outList->at(e), nullptr);
                }
            }
            return ND4J_STATUS_OK;
    }
#endif

}
}
