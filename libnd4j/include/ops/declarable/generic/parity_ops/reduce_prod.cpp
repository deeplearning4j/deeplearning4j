//
// Created by george@skymind.io on 6/1/2018.
//

#include <ops/declarable/helpers/reduce_product.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops {
#if NOT_EXCLUDED(OP_reduce_prod)

    CUSTOM_OP_IMPL(reduce_prod, 1, 1, false, 0, 0) {
        NDArray<T>* input = INPUT_VARIABLE(0);
        NDArray<T>* output = OUTPUT_VARIABLE(0);
        std::vector<int> axes = *block.getIArguments();
        const bool keepDims = block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;

        for(const auto& item : axes)
            REQUIRE_TRUE(item > -input->shapeInfo()[0] || item <input->shapeInfo()[0], 0, "REDUCE_MEAN OP: the input dimension to reduce along must be in range (-%i, %i), but got %i instead !" , input->rankOf(), input->rankOf(), item);

        input->template reduceAlongDimension<simdOps::Prod<T>>(output, axes, keepDims);

        return ND4J_STATUS_OK;
    }

    DECLARE_SHAPE_FN(reduce_prod) {    

        const bool keepDims = block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
    
        std::vector<int> dimensions = *block.getIArguments();
        Nd4jLong* outShapeInfo = ShapeUtils<T>::evalReduceShapeInfo(shape::order(inputShape->at(0)), dimensions, inputShape->at(0), keepDims, false, block.getWorkspace());

        return SHAPELIST(outShapeInfo);
    }
#endif 
#if NOT_EXCLUDED(OP_reduce_prod_bp)

    DECLARE_SHAPE_FN(reduce_prod_bp) {    

        const bool keepDims = block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
    
        Nd4jLong* outShapeInfo;
        COPY_SHAPE(inputShape->at(0), outShapeInfo);

        return SHAPELIST(outShapeInfo);
    }

    CUSTOM_OP_IMPL(reduce_prod_bp, 2, 1, false, 0, 0) {

        auto input = INPUT_VARIABLE(0);
        auto epsilon = INPUT_VARIABLE(1);
        auto output = OUTPUT_VARIABLE(0);

        const bool keepDims = block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
        T keepDimsT = (keepDims?T(1.f):T(0.f));
        // at first step we build fwd activation
        nd4j::ops::reduce_prod<T> op;
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
        auto tempProd = tmpResult->at(0);
        REQUIRE_TRUE(tempProd->isSameShape(epsilon), 0, "reduce_prod_bp: The the second param and reduce_sum output should have the equal shapes.");
    
        // tempProd has equal shape with epsilon
        if (epsilon->isScalar()) {
            auto backpropRoutine = LAMBDA_T(_x, epsilon, tempProd) {
                return (*epsilon)(0) * ((*tempProd)(0) / _x);
            };
            input->applyLambda(backpropRoutine, output);  
        } 
        else { // result 

            auto axes = *block.getIArguments();
            helpers::reduceProductBP(input, epsilon, tempProd, output, axes);
        }

        return ND4J_STATUS_OK;
    }
#endif

}
}
