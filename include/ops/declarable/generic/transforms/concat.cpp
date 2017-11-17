//
//  @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        //////////////////////////////////////////////////////////////////////////
        CUSTOM_OP_IMPL(concat, -1, 1, false, 0, 1){
            // do something here{

            int _dimension = INT_ARG(0);

            // we want to ensure that all
            NDArray<T> *first = INPUT_VARIABLE(0);
            NDArray<T> *output = this->getZ(block);

            Nd4jPointer* buffers = new Nd4jPointer[block.width()];
            Nd4jPointer* shapes = new Nd4jPointer[block.width()];

            buffers[0] = (Nd4jPointer) first->getBuffer();
            shapes[0] = (Nd4jPointer) first->getShapeInfo();

            if (nd4j::Environment::getInstance()->isDebugAndVerbose()) {
                printf("Shape %i: ", 0);
                shape::printShapeInfoLinear((int *) shapes[0]);
            }

            for (int e = 1; e < (int) block.width(); e++) {
                Variable<T> *var = block.variable(e);

                buffers[e] = (Nd4jPointer) var->getNDArray()->getBuffer();
                shapes[e] = (Nd4jPointer) var->getNDArray()->getShapeInfo();

                if (nd4j::Environment::getInstance()->isDebugAndVerbose()) {
                    printf("Shape %i: ", e);
                    shape::printShapeInfoLinear((int *) shapes[e]);
                }
            }
            if (nd4j::Environment::getInstance()->isDebugAndVerbose())
                fflush(stdout);

            nd4j::SpecialMethods<T>::concatCpuGeneric(_dimension, block.width(), buffers, shapes, output->getBuffer(), output->getShapeInfo());

            STORE_RESULT(*output);

            if (nd4j::Environment::getInstance()->isDebugAndVerbose())
                output->printShapeInfo("Concat result shape");

            delete[] buffers;
            delete[] shapes;

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(concat_v2, concat);
        DECLARE_SYN(concatv2, concat);
        DECLARE_SHAPE_FN(concat) {
            int* inp = inputShape->at(0);
            int _dimension = INT_ARG(0);

            int *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inp), int);

            std::memcpy(newShape, inp, shape::shapeInfoByteLength(inp));
            for (int i = 1; i < inputShape->size(); i++) {
                newShape[_dimension + 1] += shape::shapeOf(inputShape->at(i))[_dimension];
            }

            shape::updateStrides(newShape, shape::order(inp));

            return new ShapeList(newShape);
        }
    }
}