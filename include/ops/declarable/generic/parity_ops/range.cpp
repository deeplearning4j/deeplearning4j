//
// @author raver119@gmail.com
//

#include <NDArray.h>
#include <graph/VariableSpace.h>
#include <ops/declarable/CustomOperations.h>

/**
 * FIXME: current shape_fn design does NOT allow run-time evaluation for this op.
 */
namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(range, -2, 1, false, -2, -2) {
            std::vector<T> data;

            auto output = OUTPUT_VARIABLE(0);

            if (block.getIArguments()->size() > 0) {
                int start = INT_ARG(0);
                int stop = INT_ARG(1);
                int step = INT_ARG(2);

                int cnt = 0;
                for (int e = start; e < stop; e += step)
                    output->putScalar(cnt++, (T) e);

                STORE_RESULT(output);
            } else if (block.getTArguments()->size() > 0) {
                T start = T_ARG(0);
                T stop = T_ARG(1);
                T step = T_ARG(2);

                int cnt = 0;
                for (T e = start; e < (T) stop; e += step)
                    output->putScalar(cnt++, (T) e);

                STORE_RESULT(output);
            } else if (block.width() > 0) {
                REQUIRE_TRUE(block.width() == 3, 0, "Runtime range should have 3 arrays as input, but got %i instead", block.width());

                auto arr0 = INPUT_VARIABLE(0);
                auto arr1 = INPUT_VARIABLE(1);
                auto arr2 = INPUT_VARIABLE(2);

                T start = arr0->getScalar(0);
                T stop = arr1->getScalar(0);
                T step = arr2->getScalar(0);

                for (T e = start; e < (T) stop; e += step)
                    data.emplace_back((T) e);

                auto array = new nd4j::NDArray<T>(1, data.size(), 'c', block.getWorkspace());
                memcpy(array->buffer(), data.data(), data.size() * sizeof(T));    
    
                // we have to override existing ndarray in node variable in this case
                auto varSpace = block.getVariableSpace();
                if (varSpace->hasVariable(block.getNodeId())) {
                    auto var = varSpace->getVariable(block.getNodeId());
                    auto arr = var->getNDArray();

                    if (arr != nullptr && var->isRemovable())
                        delete arr;
                        
                    var->setNDArray(array);
                    var->markRemovable(true);
                } else {
                    varSpace->putVariable(block.getNodeId(), array);
                }
            } else {
                REQUIRE_TRUE(false, 0, "Runtime range should have inputs defined in any possible way: T_args, INT_args, or INPUT variables")
            }

            return ND4J_STATUS_OK;
        }
        DECLARE_SHAPE_FN(range) {
            int *newShape;
            std::vector<int> shape({1});
            if (block.getIArguments()->size() > 0) {
                int start = INT_ARG(0);
                int stop = INT_ARG(1);
                int step = INT_ARG(2);

                int cnt = 0;
                for (int e = start; e < stop; e += step)
                    cnt++;
                
                shape.emplace_back(cnt);
            } else if (block.getTArguments()->size() > 0) {
                T start = T_ARG(0);
                T stop = T_ARG(1);
                T step = T_ARG(2);

                int cnt = 0;
                for (T e = start; e < (T) stop; e += step)
                    cnt++;

                shape.emplace_back(cnt);
            } else {
                // FIXME:if that's runtime evaluation - we'll just pass some vector. 
                shape.emplace_back(119);
            }
            
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(2), int);
            shape::shapeBuffer(2, shape.data(), newShape);

            return new ShapeList(newShape);
        }
    }
}