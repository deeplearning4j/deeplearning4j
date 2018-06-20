//
//  @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>
#include <array>

namespace nd4j {
    namespace ops {
        //////////////////////////////////////////////////////////////////////////
        CUSTOM_OP_IMPL(concat, -1, 1, false, 0, -2){
            // do something here{
            NDArray<T> *last = INPUT_VARIABLE((int) block.width() - 1);

            int _dimension = 0;
            if (block.numI() > 0)
                _dimension = INT_ARG(0);
            else {
                _dimension = (int) last->getScalar(0);
            }

            // we want to ensure that all
            NDArray<T> *first = INPUT_VARIABLE(0);
            NDArray<T> *output = this->getZ(block);

            int elements = (int) block.width();
            bool oldScalars = first->rankOf() == 2 && first->isScalar();

            Nd4jPointer* buffers = new Nd4jPointer[elements];
            Nd4jPointer* shapes = new Nd4jPointer[elements];

            buffers[0] = (Nd4jPointer) first->getBuffer();
            shapes[0] = (Nd4jPointer) first->getShapeInfo();

            if (_dimension < 0)
                _dimension += first->rankOf();

            if (nd4j::Environment::getInstance()->isDebugAndVerbose()) {
                printf("Shape %i: ", 0);
                shape::printShapeInfoLinear((Nd4jLong *) shapes[0]);
            }

            for (int e = 1; e < elements; e++) {
                Variable<T> *var = block.variable(e);

                buffers[e] = (Nd4jPointer) var->getNDArray()->getBuffer();
                shapes[e] = (Nd4jPointer) var->getNDArray()->getShapeInfo();

                oldScalars &= shape::rank(var->getNDArray()->getShapeInfo()) == 2 && shape::isScalar(var->getNDArray()->getShapeInfo());

                if (nd4j::Environment::getInstance()->isDebugAndVerbose()) {
                    printf("Shape %i: ", e);
                    shape::printShapeInfoLinear((Nd4jLong *) shapes[e]);
                }
            }
            if (nd4j::Environment::getInstance()->isDebugAndVerbose())
                fflush(stdout);

            if (oldScalars) {
                nd4j_debug("OLD_SCALARS!\n","");
                _dimension = 1;
            }

            nd4j::SpecialMethods<T>::concatCpuGeneric(_dimension, elements, buffers, shapes, output->getBuffer(), output->getShapeInfo());

            STORE_RESULT(*output);

            if (nd4j::Environment::getInstance()->isDebugAndVerbose())
                output->printShapeInfo("Concat result shape");

            delete[] buffers;
            delete[] shapes;

            return ND4J_STATUS_OK;
        }

        DECLARE_SYN(ParallelConcat, concat);
        DECLARE_SYN(concat_v2, concat);
        DECLARE_SYN(concatv2, concat);
        
        DECLARE_SHAPE_FN(concat) {
            auto inp = inputShape->at(0);
            int _dimension = INT_ARG(0);

            auto last = inputShape->at(inputShape->size() - 1);

            Nd4jLong elements = (int) inputShape->size();
            Nd4jLong *newShape;


            { // special cases for 0D concat
                bool allScalars = true;
                bool hasScalars = false;
                for (int e = 0; e < elements; e++) {
                    auto c = inputShape->at(e);
                    allScalars &= shape::rank(c) == 0;
                    hasScalars |= shape::rank(c) == 0;
                }

                // all scalars
                if (allScalars) {
                    ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(1), Nd4jLong);

                    shape::shapeBuffer(1, &elements, newShape);
                    return SHAPELIST(newShape);
                }

                // any scalar
                if (hasScalars) {
                    ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(1), Nd4jLong);
                    Nd4jLong length = shape::length(inp);
                    for (int i = 1; i < elements; i++) {
                       length += shape::length(inputShape->at(i));
                    }

                    shape::shapeBuffer(1, &length, newShape);
                    return SHAPELIST(newShape);
                }
            }

            
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inp), Nd4jLong);

            if (_dimension < 0)
                _dimension += shape::rank(inp);

            std::memcpy(newShape, inp, shape::shapeInfoByteLength(inp));
            for (int i = 1; i < elements; i++) {
                newShape[_dimension + 1] += shape::shapeOf(inputShape->at(i))[_dimension];
            }

            shape::updateStrides(newShape, shape::order(inp));

            return SHAPELIST(newShape);
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
            auto shapeList = SHAPELIST();

            for (int e = 0; e < inputShape->size() - 1; e++) {
                auto inShape = inputShape->at(e);
                Nd4jLong* newShape;
                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inShape), Nd4jLong);
                memcpy(newShape, inShape, shape::shapeInfoByteLength(inShape));

                shapeList->push_back(newShape);
            }

            return shapeList;
        }
    }
}
