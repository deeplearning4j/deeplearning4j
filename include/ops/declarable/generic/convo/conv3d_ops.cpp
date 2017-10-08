//
// @author raver119@gmail.com
//


#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>

namespace nd4j {
    namespace ops {
        //////////////////////////////////////////////////////////////////////////
        CONFIGURABLE_OP_IMPL(conv3d, 2, 1, false, 0, 7) {
            // cubic convo

            NDArray<T> *input = block.getVariables().at(0)->getNDArray();
            NDArray<T> *weights = block.getVariables().at(1)->getNDArray();
            NDArray<T> *bias = nullptr;
            if (block.getVariables().size() == 3)
                bias = block.getVariables().at(2)->getNDArray();

            if (input->rankOf() != 5)
                return ND4J_STATUS_BAD_DIMENSIONS;

            NDArray<T> *output = this->getZ(block);

            bool biasUsed = block.getIArguments()->at(0) != 0 && bias != nullptr;
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

            // we always expect 5d
            int dimt = 2;
            int dimh = 3;
            int dimw = 4;

            Nd4jIndex nOutputPlane = weights->sizeAt(0);
            Nd4jIndex kT           = weights->sizeAt(2);
            Nd4jIndex kH           = weights->sizeAt(3);
            Nd4jIndex kW           = weights->sizeAt(4);
            Nd4jIndex inputDepth   = input->sizeAt(dimt);
            Nd4jIndex inputHeight  = input->sizeAt(dimh);
            Nd4jIndex inputWidth   = input->sizeAt(dimw);
            Nd4jIndex outputDepth  = (inputDepth - kT) / dT + 1;
            Nd4jIndex outputWidth  = (inputWidth - kW) / dW + 1;
            Nd4jIndex outputHeight = (inputHeight - kH) / dH + 1;


            REQUIRE_TRUE(output->sizeAt(0) == input->sizeAt(0) && output->sizeAt(1) == nOutputPlane && output->sizeAt(2) == outputDepth && output->sizeAt(3) == outputHeight && output->sizeAt(4) == outputWidth, 0,
                         "Expected output shape: [%i, %i, %i, %i, %i] but got [%i, %i, %i, %i, %i] instead", input->sizeAt(0), nOutputPlane, outputDepth, outputHeight, outputWidth, output->sizeAt(0), output->sizeAt(1), output->sizeAt(2), output->sizeAt(3), output->sizeAt(4));

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

        CONFIGURABLE_OP_IMPL(conv3d_bp, 3, 1, false, 0, 7) {

            return ND4J_STATUS_OK;
        }
    }
}