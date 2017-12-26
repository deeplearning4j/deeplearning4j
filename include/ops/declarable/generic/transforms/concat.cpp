//
//  @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        //////////////////////////////////////////////////////////////////////////
        CUSTOM_OP_IMPL(concat, -1, 1, false, 0, 0){
            // do something here{
            NDArray<T> *last = INPUT_VARIABLE((int) block.width() - 1);

            int _dimension = 0;
            if (block.getIArguments() > 0)
                _dimension = INT_ARG(0);
            else {
                _dimension = (int) last->getScalar(0);
            }

            // we want to ensure that all
            NDArray<T> *first = INPUT_VARIABLE(0);
            NDArray<T> *output = this->getZ(block);



            int elements = (int) block.width();
            if (last->isScalar() && !first->isScalar())
                --elements;

            Nd4jPointer* buffers = new Nd4jPointer[elements];
            Nd4jPointer* shapes = new Nd4jPointer[elements];

            buffers[0] = (Nd4jPointer) first->getBuffer();
            shapes[0] = (Nd4jPointer) first->getShapeInfo();

            if (_dimension < 0)
                _dimension += first->rankOf();

            if (nd4j::Environment::getInstance()->isDebugAndVerbose()) {
                printf("Shape %i: ", 0);
                shape::printShapeInfoLinear((int *) shapes[0]);
            }

            for (int e = 1; e < elements; e++) {
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

            nd4j::SpecialMethods<T>::concatCpuGeneric(_dimension, elements, buffers, shapes, output->getBuffer(), output->getShapeInfo());

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

            int* last = inputShape->at(inputShape->size() - 1);

            int elements = (int) inputShape->size();
            if (!shape::isScalar(inp) && shape::isScalar(last))
                --elements;

            int *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inp), int);

            if (_dimension < 0)
                _dimension += shape::rank(inp);

            std::memcpy(newShape, inp, shape::shapeInfoByteLength(inp));
            for (int i = 1; i < elements; i++) {
                newShape[_dimension + 1] += shape::shapeOf(inputShape->at(i))[_dimension];
            }

            shape::updateStrides(newShape, shape::order(inp));

            return new ShapeList(newShape);
        }


        CUSTOM_OP_IMPL(concat_bp, -1, -1, false, 0, 1) {
            auto epsilonNext = INPUT_VARIABLE(block.width() - 1);

            auto first = INPUT_VARIABLE(0);

            int axis = INT_ARG(0);

            if (axis < 0)
                axis += first->rankOf();

            int startPos = 0;
            for (int e = 0; e < block.width() - 1; e++) {
                auto originalChunk = INPUT_VARIABLE(e);
                auto epsilonChunk = OUTPUT_VARIABLE(e);
                IndicesList indices;

                int width = originalChunk->sizeAt(axis);            

                for (int e = 0; e < epsilonNext->rankOf(); e++) {
                    if (e == axis)
                        indices.push_back(NDIndex::interval(startPos, startPos + width));
                    else
                        indices.push_back(NDIndex::all());
                }

                auto subarray = epsilonNext->subarray(indices);
                epsilonChunk->assign(subarray);

                startPos += width;

                delete subarray;
            }

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(concat_bp) {
            auto shapeList = new ShapeList();

            for (int e = 0; e < inputShape->size() - 1; e++) {
                auto inShape = inputShape->at(e);
                int* newShape;
                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inShape), int);
                memcpy(newShape, inShape, shape::shapeInfoByteLength(inShape));

                shapeList->push_back(newShape);
            }

            return shapeList;
        }
    }
}