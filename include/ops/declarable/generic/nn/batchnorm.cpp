//
// Created by raver119 on 29/10/17.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        //////////////////////////////////////////////////////////////////////////
        CONFIGURABLE_OP_IMPL(batchnorm, 5, 1, true, 4, 3) {

            NDArray<T>* x = INPUT_VARIABLE(0);
            NDArray<T>* globalMeanView = INPUT_VARIABLE(1);
            NDArray<T>* globalVarView = INPUT_VARIABLE(2);
            NDArray<T>* gamma = INPUT_VARIABLE(3);
            NDArray<T>* beta = INPUT_VARIABLE(4);
            NDArray<T> *activations = this->getZ(block);
            std::vector<int> argI = *(block.getIArguments());
            std::vector<T> argT = *(block.getTArguments());
            T eps = argT[0];

            bool training = (bool)argI[0];
            bool isLockGammaBeta = (bool)argI[1];
            bool isMinibatch  = (bool)argI[2];

            NDArray<T> *mean(nullptr), *var(nullptr);
            bool deleteX = false;
            bool deleteMeanVar = false;
            if (training) {
                deleteMeanVar = true;
                switch (x->rankOf()) {
                    case 2:
                        mean = x->template reduceAlongDimension<simdOps::Mean<T>>({0});
                        var = x->template varianceAlongDimension<simdOps::SummaryStatsVariance<T>>(false, {0});
                        break;
                    case 4:
                        mean = x->template reduceAlongDimension<simdOps::Mean<T>>({0,2,3});
                        var = x->template varianceAlongDimension<simdOps::SummaryStatsVariance<T>>(false, {0,2,3});
                        break;
                    default:
                        throw "Graph operation batchnorm: the rank of input array must be equal to 2 or 4 !";
                }
                var->template applyScalar<simdOps::Add<T>>(eps, nullptr);
            }
            else {
                mean = INPUT_VARIABLE(1);
                var = INPUT_VARIABLE(2);
            }

            NDArray<T> std(var->getShapeInfo(), block.getWorkspace());
            var->template applyTransform<simdOps::Sqrt<T>>(&std, nullptr);
            NDArray<T> xMu(x->getShapeInfo(), block.getWorkspace());
            NDArray<T> xHat(x->getShapeInfo(), block.getWorkspace());

            if (x->rankOf() == 2) {
                x->subRowVector(mean, &xMu);
                xMu.divRowVector(&std, &xHat);

                if (isLockGammaBeta) {
                    T g = argT[1];
                    T b = argT[2];
                    if (g != (T)1. && b != (T)0.) {
                        xHat.template applyScalar<simdOps::Multiply<T>>(g, activations, nullptr);
                        activations->template applyScalar<simdOps::Add<T>>(b, nullptr);
                    }
                    else
                        *activations = xHat;
                }
                else
                    xHat.mulRowVector(gamma, activations);

            }
            else if (x->rankOf() == 4) {

                if (!shape::strideDescendingCAscendingF(x->getShapeInfo())) {
                    x = x->dup(x->ordering());
                    deleteX = true;
                }

                x->template applyBroadcast<simdOps::Subtract<T>>({1}, mean, &xMu, nullptr);
                xMu.template applyBroadcast<simdOps::Divide<T>>({1}, &std, &xHat, nullptr);

                if (isLockGammaBeta) {
                    T g = argT[1];
                    T b = argT[2];
                    if (g != (T)1. && b != (T)0.) {
                        xHat.template applyScalar<simdOps::Multiply<T>>(g, activations, nullptr);
                        activations->template applyScalar<simdOps::Add<T>>(b, nullptr);
                    }
                    else
                        *activations = xHat;
                }
                else {
                    xHat.template applyBroadcast<simdOps::Multiply<T>>({1}, gamma, activations, nullptr);
                    activations->template applyBroadcast<simdOps::Add<T>>({1}, beta, activations, nullptr);
                }
            }
            else
                throw "Graph operation batchnorm: the layer prior to BatchNorm in the configuration is not currently supported !";

            T decay;
            if (training) {
                if (isMinibatch) {
                    decay = argT[3];

                    globalMeanView->template  applyScalar<simdOps::Multiply<T>>(decay, nullptr);
                    mean->template applyScalar<simdOps::Multiply<T>>((T)1. - decay, nullptr);
                    globalMeanView->template applyPairwiseTransform<simdOps::Add<T>>(mean, nullptr);

                    globalVarView->template  applyScalar<simdOps::Multiply<T>>(decay, nullptr);
                    var->template applyScalar<simdOps::Multiply<T>>((T)1. - decay, nullptr);
                    globalVarView->template applyPairwiseTransform<simdOps::Add<T>>(var, nullptr);
                }
                else {
                    globalMeanView->assign(mean);
                    globalVarView->assign(var);
                }
            }

            STORE_RESULT(*activations);

            if(deleteX)
                delete x;
            if(deleteMeanVar) {
                delete mean;
                delete var;
            }
            return ND4J_STATUS_OK;
        }

        //////////////////////////////////////////////////////////////////////////
        CONFIGURABLE_OP_IMPL(batchnorm_bp, 5, 1, true, 0, 1) {

            NDArray<T>* input = INPUT_VARIABLE(0);
            NDArray<T>* epsilon = INPUT_VARIABLE(1);
            NDArray<T>* gamma = INPUT_VARIABLE(2);
            NDArray<T>* dGlobalMeanView = INPUT_VARIABLE(3);
            NDArray<T>* dGlobalVarView = INPUT_VARIABLE(4);
            NDArray<T>* outEpsilon = this->getZ(block);
            std::vector<int> argI = *(block.getIArguments());
            const int bS = epsilon->sizeAt(0);
            bool isLockGammaBeta = (bool)argI[0];
            const int* epsilonShape = epsilon->getShapeInfo() + 1;
            const T eps = (T)1e-5;

            int rank = epsilon->rankOf();
            std::initializer_list<int> dimensions;
            int effectiveBatchSize;
            if (rank == 2) {
                dimensions = {0};
                effectiveBatchSize = bS;
            }
            else if (rank == 4) {
                dimensions = {0, 2, 3};
                effectiveBatchSize = input->sizeAt(0)*input->sizeAt(2)*input->sizeAt(3);
            }
            else
                throw "Graph operation batchnorm_bp: the epsilon rank must be equal to 2 or 4 !";

            NDArray<T> *mean(nullptr), *var(nullptr), *dBeta(nullptr), *dGamma(nullptr), *dLdVar(nullptr), *dxmu1(nullptr), *dxmu2(nullptr);
            mean = input->template reduceAlongDimension<simdOps::Mean<T>>(dimensions);
            var = input->template varianceAlongDimension<simdOps::SummaryStatsVariance<T>>(false, dimensions);
            var->template applyScalar<simdOps::Add<T>>(eps, nullptr);
            auto std = new NDArray<T>(var->getShapeInfo(), block.getWorkspace());
            var->template applyTransform<simdOps::Sqrt<T>>(std, nullptr);

            auto xMu = new NDArray<T>(input->getShapeInfo(), block.getWorkspace());
            auto xHat = new NDArray<T>(input->getShapeInfo(), block.getWorkspace());
            auto temp1 = new NDArray<T>(epsilon->getShapeInfo(), block.getWorkspace());
            auto temp2 = new NDArray<T>(std->getShapeInfo(), block.getWorkspace());
            auto dGammaView = new NDArray<T>('c', {1, epsilonShape[1]}, block.getWorkspace());
            auto dBetaView = new NDArray<T>('c', {1, epsilonShape[1]}, block.getWorkspace());
            auto dxhat = new NDArray<T>(epsilon->getShapeInfo(), block.getWorkspace());

            if (rank == 2) {
                input->subRowVector(mean, xMu);
                xMu->divRowVector(std, xHat);
            }
            else {
                input->template applyBroadcast<simdOps::Subtract<T>>({1}, mean, xMu, nullptr);
                xMu->template applyBroadcast<simdOps::Divide<T>>({1}, std, xHat, nullptr);
            }

            dBeta = epsilon->sum(dimensions); // dL/dBeta = sum_examples dL/dOut
            epsilon->template applyPairwiseTransform<simdOps::Multiply<T>>(xHat, temp1, nullptr);   //dL/dGamma = sum_examples dL/dOut .* xHat
            dGamma = temp1->sum(dimensions);  //dL/dGamma = sum_examples dL/dOut .* xHat

            if (isLockGammaBeta)
                epsilon->template applyPairwiseTransform<simdOps::Multiply<T>>(gamma, dxhat, nullptr);
            else {// Standard case
                if(rank == 2)
                    epsilon->mulRowVector(gamma, dxhat); //dL/dxHat = dL/dOut . gamma        Shape: [minibatchSize, nOut]
                else
                    epsilon->template applyBroadcast<simdOps::Multiply<T>>({1}, gamma, dxhat, nullptr);
            }

            // dLdVar - dL/dVariance, shape: [1, miniBatch]
            dxhat->template applyPairwiseTransform<simdOps::Multiply<T>>(xMu, temp1, nullptr);
            dLdVar = temp1->sum(dimensions);
            dLdVar->template applyScalar<simdOps::Multiply<T>>((T)-0.5, nullptr);
            T powParams[] = {(T)(-3.)};
            std->template applyTransform<simdOps::Pow<T>>(temp2, powParams);
            dLdVar->template applyPairwiseTransform<simdOps::Multiply<T>>(temp2, nullptr);

            //dL/dmu
            dxmu1 = dxhat->sum(dimensions);
            dxmu1->template applyPairwiseTransform<simdOps::Divide<T>>(std, nullptr);
            dxmu1->template applyTransform<simdOps::Neg<T>>();
            dxmu2 = xMu->sum(dimensions);
            dxmu2->template applyScalar<simdOps::Multiply<T>>((T)(-2.)/effectiveBatchSize);
            dxmu2->template applyPairwiseTransform<simdOps::Multiply<T>>(dLdVar, nullptr);

            dxmu1->template applyPairwiseTransform<simdOps::Add<T>>(dxmu2, nullptr);
            NDArray<T>* dLdmu = dxmu1;      //  = dL/dmu Shape: [1, nOut]

            //Note the array reuse here: dxhat, xMu, dLdVar, dLdmu - all are invalid after this line (but aren't used later anyway)
            NDArray<T>* dLdx = dxhat;
            dLdVar->template applyScalar<simdOps::Multiply<T>>((T)(2.)/effectiveBatchSize);
            dLdmu->template applyScalar<simdOps::Multiply<T>>((T)(1.)/effectiveBatchSize);
            if(rank == 2) {
                dLdx->divRowVector(std, dLdx);
                xMu->mulRowVector(dLdVar, xMu);
            }
            else {
                dLdx->template applyBroadcast<simdOps::Divide<T>>({1}, std, dLdx, nullptr);
                xMu->template applyBroadcast<simdOps::Multiply<T>>({1}, dLdVar, xMu, nullptr);
            }
            dLdx->template applyPairwiseTransform<simdOps::Add<T>>(xMu, nullptr);
            if(rank == 2)
                dLdx->addRowVector(dLdmu, dLdx);
            else
                dLdx->template applyBroadcast<simdOps::Add<T>>({1}, dLdmu, dLdx, nullptr);

            *outEpsilon = *dLdx;

            //TODO rework this to avoid the assign here
            // dGammaView->assign(dGamma);
            // dBetaView->assign(dBeta);
            // dGlobalMeanView->assign((T)0.);
            // dGlobalVarView->assign((T)0.);
            // retGradient.setGradientFor(BatchNormalizationParamInitializer.GAMMA, dGammaView);
            // retGradient.setGradientFor(BatchNormalizationParamInitializer.BETA, dBetaView);
            // retGradient.setGradientFor(BatchNormalizationParamInitializer.GLOBAL_MEAN, dGlobalMeanView);
            // retGradient.setGradientFor(BatchNormalizationParamInitializer.GLOBAL_VAR, dGlobalVarView);

            delete std;
            delete xMu;
            delete xHat;
            delete mean;
            delete var;
            delete dBeta;
            delete dGamma;
            delete dLdVar;
            delete dxmu1;
            delete dxmu2;
            delete temp1;
            delete temp2;
            delete dxhat;
            delete dGammaView;
            delete dBetaView;

            return ND4J_STATUS_OK;
        }
    }
}