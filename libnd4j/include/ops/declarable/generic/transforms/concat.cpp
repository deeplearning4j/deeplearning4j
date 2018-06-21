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
            NDArray<T> *first = nullptr;
            auto output = OUTPUT_VARIABLE(0);

            int elements = 0;

            for (int e = 0; e < block.width(); e++) {
                auto arr = INPUT_VARIABLE(e);
                if (!arr->isEmpty())
                    elements++;

                // we must find first non-empty element here
                if (!arr->isEmpty() && first == nullptr)
                    first = arr;
            }

            REQUIRE_TRUE(first != nullptr, 0, "Concat: at least 1 non-empty input required!");

            // it's possible to get into situation when your input has only 1 input. That's just assign
            if (elements == 1) {
                output->assign(first);
                return Status::OK();
            }

            bool oldScalars = first->rankOf() == 2 && first->isScalar();

            auto buffers = new Nd4jPointer[elements];
            auto shapes = new Nd4jPointer[elements];

            buffers[0] = (Nd4jPointer) first->getBuffer();
            shapes[0] = (Nd4jPointer) first->getShapeInfo();

            if (_dimension < 0)
                _dimension += first->rankOf();

            if (nd4j::Environment::getInstance()->isDebugAndVerbose()) {
                printf("Shape %i: ", 0);
                shape::printShapeInfoLinear((Nd4jLong *) shapes[0]);
            }

            int er = 0;
            for (int e = 0; e < block.width(); e++) {
                Variable<T> *var = block.variable(e);
                auto array = var->getNDArray();

                if (array->isEmpty())
                    continue;

                buffers[er] = reinterpret_cast<Nd4jPointer>(array->getBuffer());
                shapes[er++] = reinterpret_cast<Nd4jPointer>(array->getShapeInfo());

                oldScalars &= array->rankOf() == 2 && array->isScalar();

                if (nd4j::Environment::getInstance()->isDebugAndVerbose()) {
                    printf("Shape %i: ", e);
                    shape::printShapeInfoLinear(array->shapeInfo());
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

            NDArray<T>* first = nullptr;
            auto last = inputShape->at(inputShape->size() - 1);

            Nd4jLong elements = 0;
            Nd4jLong *newShape;

            for (int  e = 0; e < inputShape->size(); e++) {
                auto s = INPUT_VARIABLE(e);

                if (!s->isEmpty()) {
                    elements++;

                    if (first == nullptr)
                        first = s;
                }
            }


            { // special cases for 0D concat
                bool allScalars = true;
                bool hasScalars = false;
                for (int e = 0; e < block.width(); e++) {
                    auto c = INPUT_VARIABLE(e);

                    if (c->isEmpty())
                        continue;

                    allScalars &= c->rankOf() == 0;
                    hasScalars |= c->rankOf() == 0;
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
                    for (int i = 1; i < block.width(); i++) {
                        auto c = INPUT_VARIABLE(i);
                        if (c->isEmpty())
                            continue;

                        length += c->lengthOf();
                    }

                    shape::shapeBuffer(1, &length, newShape);
                    return SHAPELIST(newShape);
                }
            }

            
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(first->shapeInfo()), Nd4jLong);

            if (_dimension < 0)
                _dimension += first->rankOf();

            std::memcpy(newShape, first->shapeInfo(), shape::shapeInfoByteLength(first->shapeInfo()));
            for (int i = 0; i < inputShape->size(); i++) {
                auto s = INPUT_VARIABLE(i);

                // FIXME: s == first is bad, but fast. alternatively we can subtract first size out of result
                if (s->isEmpty() || s == first)
                    continue;

                newShape[_dimension + 1] += shape::shapeOf(inputShape->at(i))[_dimension];
            }

            shape::updateStrides(newShape, first->ordering());

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
