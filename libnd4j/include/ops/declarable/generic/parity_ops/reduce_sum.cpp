//
// Created by george@skymind.io on 6/1/2018.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops {
#if NOT_EXCLUDED(OP_reduce_sum)

    CUSTOM_OP_IMPL(reduce_sum, 1, 1, false, 0, 0) {
        NDArray<T>* input = INPUT_VARIABLE(0);
        NDArray<T>* output = OUTPUT_VARIABLE(0);
        std::vector<int> axes = *block.getIArguments();

        for(const auto& item : axes)
            REQUIRE_TRUE(item > -input->shapeInfo()[0] || item <input->shapeInfo()[0], 0, "REDUCE_MEAN OP: the input dimension to reduce along must be in range (-%i, %i), but got %i instead !" , input->rankOf(), input->rankOf(), item);

        const bool keepDims = block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
        input->template reduceAlongDimension<simdOps::Sum<T>>(output, axes, keepDims);

        return ND4J_STATUS_OK;
    }

    DECLARE_SHAPE_FN(reduce_sum) {    

        const bool keepDims = block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
    
        std::vector<int> dimensions = *block.getIArguments();
        Nd4jLong* outShapeInfo = ShapeUtils<T>::evalReduceShapeInfo(shape::order(inputShape->at(0)), dimensions, inputShape->at(0), keepDims, false, block.getWorkspace());

        return SHAPELIST(outShapeInfo);
    }
#endif 
#if NOT_EXCLUDED(OP_reduce_sum_bp)

    DECLARE_SHAPE_FN(reduce_sum_bp) {    

        const bool keepDims = block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
    
        //std::vector<int> dimensions = *block.getIArguments();
        Nd4jLong* outShapeInfo;// = ShapeUtils<T>::evalReduceShapeInfo(shape::order(inputShape->at(0)), dimensions, inputShape->at(0), keepDims, false, block.getWorkspace());
        COPY_SHAPE(inputShape->at(0), outShapeInfo);
        return SHAPELIST(outShapeInfo);
    }

    CUSTOM_OP_IMPL(reduce_sum_bp, 2, 1, false, 0, 0) {

            auto input = INPUT_VARIABLE(0);
            auto epsilon = INPUT_VARIABLE(1);
            auto output = OUTPUT_VARIABLE(0);

            //REQUIRE_TRUE(output->isSameShape(epsilon), 0, "reduce_sum_bp: The second param shape should be the same as result shape.");
            if (epsilon->isScalar()) {
                output->assign(epsilon);
            }
            else {
                auto axes = *block.getIArguments();
//                std::unique_ptr<ResultSet<T>> outList(NDArrayFactory<T>::allTensorsAlongDimension(output, dimensions));
                std::vector<int> dimensions; //(input->rankOf() - axes.size());
                for (Nd4jLong e = 0; e < input->rankOf(); e++) {
                    if (std::find(axes.begin(), axes.end(), e) == axes.end()) {
                        dimensions.emplace_back(e);
                    }
                }
                std::unique_ptr<ResultSet<T>> outList(NDArrayFactory<T>::allTensorsAlongDimension(output, dimensions));
                //output->
                for (Nd4jLong e = 0; e < outList->size(); ++e) {
                    outList->at(e)->assign(epsilon);
                }
            }

            return ND4J_STATUS_OK;
    }
#endif

}
}
