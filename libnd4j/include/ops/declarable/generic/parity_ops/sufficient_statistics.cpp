//
// Created by george@skymind.io on 2/21/2018.
// Modified by sgazeos@gmail.com on 4/4/2018

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_sufficient_statistics)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/axis.h>
namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(sufficient_statistics, 2, 3, false, 0, 0) {
            NDArray<T>* input = INPUT_VARIABLE(0);
            NDArray<T>* axisVector = INPUT_VARIABLE(1);
            NDArray<T>* dataCount = OUTPUT_VARIABLE(0);

            NDArray<T>* sum = OUTPUT_VARIABLE(1);
            NDArray<T>* squares = OUTPUT_VARIABLE(2);

            std::vector<int> axis(axisVector->lengthOf());//*block.getIArguments();

            // axis might be dynamic (i.e. tf mode)
            helpers::adjustAxis(input, axisVector, axis);

            input->template reduceAlongDimension<simdOps::SquaredNorm<T>>(squares, axis);
            input->template reduceAlongDimension<simdOps::Sum<T>>(sum, axis);
            dataCount->putScalar(0, input->lengthOf() / sum->lengthOf());
            if (block.numT() > 0) {
                NDArray<T>* shift = OUTPUT_VARIABLE(3);
                shift->putScalar(0, T_ARG(0));
            }

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(sufficient_statistics) {

            NDArray<T>* axisVector = INPUT_VARIABLE(1);
            std::vector<int> axis(axisVector->lengthOf());

            NDArray<T>* input = INPUT_VARIABLE(0);

            for (int e = 0; e < axisVector->lengthOf(); e++) {
                int ca = (int) axisVector->getScalar(e);
                if (ca < 0)
                        ca += input->rankOf();

                    axis[e] = ca;
            }
            //std::vector<int> dims = ShapeUtils<T>::convertAxisToTadTarget(input->rankOf(), {axis});
            auto scalarShape = ShapeUtils<T>::createScalarShapeInfo(block.workspace());
            auto sumShape = ShapeUtils<T>::evalReduceShapeInfo('c', axis, *input, false, false, block.workspace());
            auto squareShape = ShapeUtils<T>::evalReduceShapeInfo('c', axis, *input, false, false, block.workspace());
            auto shapeList = SHAPELIST(scalarShape, sumShape, squareShape); 
            if (block.numT() > 0)
                shapeList->push_back(ShapeUtils<T>::createScalarShapeInfo(block.workspace()));
            
            return shapeList;
        }
    }

}

#endif