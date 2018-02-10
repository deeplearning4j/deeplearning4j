//
//  @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>
#include <declarable/generic/helpers/convolutions.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(conv1d, 2, 1, false, 0, 3) {
            auto input = INPUT_VARIABLE(0);
            auto weights = INPUT_VARIABLE(1);
            NDArray<T>* bias = nullptr;
            if (block.width() > 2)
                bias = INPUT_VARIABLE(2);

            int kernel = INT_ARG(0);
            int stride = INT_ARG(1);
            int padding = INT_ARG(2);
            bool isSameMode = false;
            if (block.getIArguments()->size() > 3)
                isSameMode = (bool) INT_ARG(3);

            REQUIRE_TRUE(weights->rankOf() == 3, 0, "Conv1D requires 3D input, but got %iD instead", input->rankOf());
            REQUIRE_TRUE(input->rankOf() == 3, 0, "Conv1D requires 3D input, but got %iD instead", input->rankOf());
            
            auto z = OUTPUT_VARIABLE(0);

            auto _input = input->reshape(input->ordering(),{input->sizeAt(0), input->sizeAt(1), input->sizeAt(2), 1});
            auto _weights = weights->reshape(weights->ordering(),{weights->sizeAt(0), weights->sizeAt(1), weights->sizeAt(2), 1});

            nd4j::ops::conv2d<T> op;
            auto result = op.execute({_input, _weights, bias}, {}, {kernel, 1, stride, 1, padding, 0, 1, 1, isSameMode, 0});
            auto tmpZ = result->at(0);

            tmpZ->reshapei(tmpZ->ordering(), {tmpZ->sizeAt(0), tmpZ->sizeAt(1), tmpZ->sizeAt(2)});

            z->assign(tmpZ);

            delete result;
            delete _input;
            delete _weights;
            
            return ND4J_STATUS_OK;
        }
        DECLARE_SHAPE_FN(conv1d) {
            auto inShape = inputShape->at(0);
            auto wShape = inputShape->at(1);

            const int kY = INT_ARG(0);
            const int sY = INT_ARG(1);
            int pY = INT_ARG(2);
            int pX = 0;
            bool isSameMode = false;
            if (block.getIArguments()->size() > 3)
                isSameMode = (bool) INT_ARG(3);

            int oY = 0;
            int oX = 0;

            const int batchSize = inShape[1];
            const int outDepth = wShape[1];
            const int inY = inShape[3];
            const int inX = 1; // constant value

            ConvolutionUtils<T>::calcOutSizePool2D(oY, oX, kY, 1, sY, 1, pY, pX, 1, 1, inY, inX, isSameMode);

            if (isSameMode)
                ConvolutionUtils<T>::_calcPadding2D(pY, pX, oY, oX, inY, inX, kY, 1, sY, 1, 1, 1);

            int *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(3), int);
            std::vector<int> shape({batchSize, outDepth, oY});
            shape::shapeBuffer(3, shape.data(), newShape);

            return new ShapeList(newShape);
        }

        CUSTOM_OP_IMPL(conv1d_bp, 3, 2, false, 0, 3) {
            NDArray<T>* input = INPUT_VARIABLE(0);
            NDArray<T>* weights = INPUT_VARIABLE(1);
            NDArray<T>* bias = nullptr;
            NDArray<T>* epsilonNext;

            NDArray<T>* epsilon = OUTPUT_VARIABLE(0);
            NDArray<T>* gradW = OUTPUT_VARIABLE(1);
            NDArray<T>* gradB = nullptr;
            if (block.width() == 3)
                epsilonNext = INPUT_VARIABLE(2);
            else {
                bias = INPUT_VARIABLE(2);
                epsilonNext = INPUT_VARIABLE(3);
                gradB = OUTPUT_VARIABLE(2);
            }

            REQUIRE_TRUE(input->rankOf() == 3, 0, "Conv1D expects 3D input, but got %i instead", input->rankOf());
            REQUIRE_TRUE(weights->rankOf() == 3, 0, "Conv1D expects 3D weights, but got %i instead", weights->rankOf());
            REQUIRE_TRUE(epsilonNext->rankOf() == 3, 0, "Conv1D expects 3D epsilons, but got %i instead", epsilonNext->rankOf());

            int kernel = INT_ARG(0);
            int stride = INT_ARG(1);
            int padding = INT_ARG(2);
            bool isSameMode = false;
            if (block.getIArguments()->size() > 3)
                isSameMode = (bool) INT_ARG(3);

            auto _input = input->reshape(input->ordering(),{input->sizeAt(0), input->sizeAt(1), input->sizeAt(2), 1});
            auto _weights = weights->reshape(weights->ordering(),{weights->sizeAt(0), weights->sizeAt(1), weights->sizeAt(2), 1});
            auto _epsilonNext = epsilonNext->reshape(epsilonNext->ordering(),{epsilonNext->sizeAt(0), epsilonNext->sizeAt(1), epsilonNext->sizeAt(2), 1});

            nd4j::ops::conv2d_bp<T> op;

            if (bias != nullptr) {
                auto result = op.execute({_input, _weights, bias, _epsilonNext}, {}, {kernel, 1, stride, 1, padding, 0, 1, 1, isSameMode});

                auto tmpEps = result->at(0);
                auto tmpGW = result->at(1);
                auto tmpGB = result->at(2);

                tmpEps->reshapei(tmpEps->ordering(), {tmpEps->sizeAt(0), tmpEps->sizeAt(1), tmpEps->sizeAt(2)});
                tmpGW->reshapei(tmpGW->ordering(), {tmpGW->sizeAt(0), tmpGW->sizeAt(1), tmpGW->sizeAt(2)});

                epsilon->assign(tmpEps);
                gradW->assign(tmpGW);
                gradB->assign(tmpGB);

                delete result;
            } else {
                auto result = op.execute({_input, _weights, epsilonNext}, {}, {kernel, 1, stride, 1, padding, 0, 1, 1, isSameMode});

                auto tmpEps = result->at(0);
                auto tmpGW = result->at(1);

                tmpEps->reshapei(tmpEps->ordering(), {tmpEps->sizeAt(0), tmpEps->sizeAt(1), tmpEps->sizeAt(2)});
                tmpGW->reshapei(tmpGW->ordering(), {tmpGW->sizeAt(0), tmpGW->sizeAt(1), tmpGW->sizeAt(2)});

                epsilon->assign(tmpEps);
                gradW->assign(tmpGW);

                delete result;
            }

            delete _weights;
            delete _input;
            delete _epsilonNext;


            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(conv1d_bp) {
            auto inShape = inputShape->at(0);
            auto wShape = inputShape->at(1);

            int *bShape = nullptr;
            // if conv1d op has bias provided, we'll have > 3 inputs (input, weights, _bias_, epsilonNext)
            if (inputShape->size() > 3)
                bShape = inputShape->at(2);

            int *newIShape;
            ALLOCATE(newIShape, block.getWorkspace(), shape::shapeInfoLength(inShape), int);
            memcpy(newIShape, inShape, shape::shapeInfoByteLength(inShape));

            int *newWShape;
            ALLOCATE(newWShape, block.getWorkspace(), shape::shapeInfoLength(wShape), int);
            memcpy(newWShape, wShape, shape::shapeInfoByteLength(wShape));

            auto shapeList = new ShapeList({newIShape, newWShape});

            if (bShape != nullptr) {
                int *newBShape;
                ALLOCATE(newBShape, block.getWorkspace(), shape::shapeInfoLength(bShape), int);
                memcpy(newBShape, bShape, shape::shapeInfoByteLength(bShape));

                shapeList->push_back(newBShape);
            }

            return shapeList;
        }
    }
}