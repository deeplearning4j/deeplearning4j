//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_squeeze)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(squeeze, 1, 1, true, 0, -2) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            std::vector<int> axis;

            if (block.numI() > 0)
                for (int e = 0; e < block.numI(); e++) {
                    int _a = INT_ARG(e);
                    if (_a < 0)
                        _a += input->rankOf();
                        
                    axis.emplace_back(_a);
                }
            else if (block.width() > 1) {
                auto a = INPUT_VARIABLE(1);
                for (int e = 0; e < a->lengthOf(); e++) {
                    int _a = (int) a->getScalar(e);
                    
                    if (_a < 0)
                        _a += input->rankOf();

                    axis.emplace_back(_a);
                }
            }

            if (input->rankOf() == 0 || (input->rankOf() == 1 && input->lengthOf() == 1)) {
                output->assign(input->buffer()[0]);
                return ND4J_STATUS_OK;
            }

            std::vector<Nd4jLong> shape;
            if (axis.size() == 0) {
                for (int d = 0; d < input->rankOf(); d++)
                    if (input->sizeAt(d) > 1)
                        shape.emplace_back(input->sizeAt(d));
            } else {
                for (int d = 0; d < input->rankOf(); d++) {
                    if (input->sizeAt(d) == 1) {
                        if (std::find(axis.begin(), axis.end(), d) == axis.end())
                            shape.emplace_back(input->sizeAt(d));
                    } else shape.emplace_back(input->sizeAt(d));
                }
            }

            if (block.isInplace()) {
                output->reshapei(input->ordering(), shape);
            } else {
                auto tmp = input->reshape(input->ordering(), shape);
                output->assign(tmp);
                delete tmp;
            }

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(squeeze) {
            auto shapeList = SHAPELIST();

            Nd4jLong* newShape;
            auto in = inputShape->at(0);
            auto rank = shape::rank(in);
            auto length = shape::length(in);

            if (rank == 0 || (rank == 1 && length == 1)) {
                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(0), Nd4jLong);
                newShape[0] = 0;
                newShape[1] = 0;
                newShape[2] = 1;
                newShape[3] = 99;

                shapeList->push_back(newShape);
                return shapeList;
            }

            std::vector<int> axis;

            if (block.numI() > 0)
                for (int e = 0; e < block.numI(); e++) {
                    int _a = INT_ARG(e);
                    if (_a < 0)
                        _a += rank;
                        
                    axis.emplace_back(_a);
                }
            else if (block.width() > 1) {
                auto a = INPUT_VARIABLE(1);
                for (int e = 0; e < a->lengthOf(); e++) {
                    int _a = (int) a->getScalar(e);
                    
                    if (_a < 0)
                        _a += rank;

                    axis.emplace_back(_a);
                }
                
            }

            auto order = shape::order(in);
            auto oldShape = shape::shapeOf(in);

            std::vector<Nd4jLong> shape;
            if (axis.size() == 0) {
                for (int d = 0; d < rank; d++)
                    if (oldShape[d] > 1)
                        shape.emplace_back(oldShape[d]);
            } else {
                for (int d = 0; d < rank; d++) {
                    if (oldShape[d] == 1) {
                        if (std::find(axis.begin(), axis.end(), d) == axis.end())
                            shape.emplace_back(oldShape[d]);
                    } else shape.emplace_back(oldShape[d]);
                }
            }

            if (shape.size() == 0) {
                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(0), Nd4jLong);
                newShape[0] = 0;
                newShape[1] = 0;
                newShape[2] = 1;
                newShape[3] = 99;

                shapeList->push_back(newShape);
                return shapeList;
            }

            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(shape.size()), Nd4jLong);
            if (order == 'c')
                shape::shapeBuffer(shape.size(), shape.data(), newShape);
            else
                shape::shapeBufferFortran(shape.size(), shape.data(), newShape);

            shapeList->push_back(newShape);

            return shapeList;
        }
    }
}

#endif