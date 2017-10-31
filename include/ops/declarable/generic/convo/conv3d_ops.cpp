//
// @author raver119@gmail.com
//


#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>

namespace nd4j {
    namespace ops {
        //////////////////////////////////////////////////////////////////////////
        CUSTOM_OP_IMPL(conv3d, 2, 1, false, 0, 7) {
            // cubic convo

            NDArray<T> *input = INPUT_VARIABLE(0);
            NDArray<T> *weights = INPUT_VARIABLE(1);
            NDArray<T> *bias = nullptr;
            if (block.width() == 3)
                bias = INPUT_VARIABLE(2);

            if (input->rankOf() != 5)
                return ND4J_STATUS_BAD_DIMENSIONS;

            NDArray<T>* output = OUTPUT_VARIABLE(0);                

            bool biasUsed = block.getIArguments()->at(0) != 0 && bias != nullptr;
            // TODO change width/height order  height/width
            int dT = block.getIArguments()->at(1);
            int dW = block.getIArguments()->at(2);
            int dH = block.getIArguments()->at(3);
            int pT = block.getIArguments()->at(4);
            int pW = block.getIArguments()->at(5);
            int pH = block.getIArguments()->at(6);


            if (pT != 0 || pW != 0 || pH != 0) {
                nd4j_printf("Padding isn't supported on CPU backend O_o","");
                return ND4J_STATUS_BAD_PARAMS;
            }
            
            std::unique_ptr<ArrayList<T>> batchIn(NDArrayFactory<T>::allExamples(input));
            std::unique_ptr<ArrayList<T>> batchOut(NDArrayFactory<T>::allExamples(output));

            // TODO: eventually we want OMP being used here
            for (int e = 0; e < batchIn->size(); e++) {
                auto tadIn = batchIn->at(e);
                auto tadOut = batchOut->at(e);

                if (biasUsed) {
                    std::unique_ptr<ArrayList<T>> outputBlock(NDArrayFactory<T>::allExamples(tadOut));
                    for (int i = 0; i < bias->lengthOf(); i++) {
                        auto oB = outputBlock->at(i);
                        oB->assign(bias->getScalar(i));
                    }
                } else
                    output->assign(0.0);

                Nd4jStatus  res = ConvolutionUtils<T>::conv3Dmv(tadOut, (T) 1.0f, (T) 1.0f, tadIn, weights, dT, dH, dW, "V", "X");
                if (res != ND4J_STATUS_OK)
                    throw "Boom";
            }

            STORE_RESULT(*output);

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(conv3d) {
 
            // REQUIRE_TRUE(output->sizeAt(0) == input->sizeAt(0) && output->sizeAt(1) == nOutputPlane && output->sizeAt(2) == outputDepth && output->sizeAt(3) == outputHeight && output->sizeAt(4) == outputWidth, 0,
            // "Expected output shape: [%i, %i, %i, %i, %i] but got [%i, %i, %i, %i, %i] instead", input->sizeAt(0), nOutputPlane, outputDepth, outputHeight, outputWidth, output->sizeAt(0), output->sizeAt(1), output->sizeAt(2), output->sizeAt(3), output->sizeAt(4));

            int* inputShapeInfo  = inputShape->at(0);
            int* weightShapeInfo = inputShape->at(1);
            
            int rank         = inputShapeInfo[0];              // = 5
            int bS           = inputShapeInfo[1];            
            int inputDepth   = inputShapeInfo[3];
            int inputHeight  = inputShapeInfo[4];
            int inputWidth   = inputShapeInfo[5];
            int nOutputPlane = weightShapeInfo[1];
            int kT           = weightShapeInfo[3];
            int kH           = weightShapeInfo[4];
            int kW           = weightShapeInfo[5];
            int dT           = block.getIArguments()->at(1);
            int dW           = block.getIArguments()->at(2);
            int dH           = block.getIArguments()->at(3);

            int outputDepth  = (inputDepth  - kT) / dT + 1;
            int outputHeight = (inputHeight - kH) / dH + 1;
            int outputWidth  = (inputWidth  - kW) / dW + 1;

            int shapeInfoLength = rank*2 + 4;        
            char order = (char)(inputShapeInfo[shapeInfoLength-1]);
        
            int* newShapeInfo = nullptr;
            ALLOCATE(newShapeInfo, block.getWorkspace(), shapeInfoLength, int);

            newShapeInfo[0] = rank;
            newShapeInfo[1] = bS;
            newShapeInfo[2] = nOutputPlane;
            newShapeInfo[3] = outputDepth;
            newShapeInfo[4] = outputHeight;
            newShapeInfo[5] = outputWidth;

            shape::updateStrides(newShapeInfo, order);

            return new ShapeList(newShapeInfo);
        }

        //////////////////////////////////////////////////////////////////////////
        CONFIGURABLE_OP_IMPL(conv3d_bp, 3, 1, false, 0, 7) {

            return ND4J_STATUS_OK;
        }
    }
}