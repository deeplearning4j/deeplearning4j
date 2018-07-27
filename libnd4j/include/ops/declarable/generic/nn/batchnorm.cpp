/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// Created by raver119 on 29/10/17.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_batchnorm)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops {
CUSTOM_OP_IMPL(batchnorm, 5, 1, false, 1, 2) {

    NDArray<T>* input    = INPUT_VARIABLE(0);
    NDArray<T>* mean     = INPUT_VARIABLE(1);
    NDArray<T>* variance = INPUT_VARIABLE(2);
    NDArray<T>* gamma    = INPUT_VARIABLE(3);
    NDArray<T>* beta     = INPUT_VARIABLE(4);
    
    NDArray<T>* output   = OUTPUT_VARIABLE(0);

    // check whether all input shapes are mutually broadcastable 
    // if yes, evaluate output shapeInfo which is common broadcast shape for all input arrays
    REQUIRE_TRUE(output->getShapeInfo() != nullptr, 0, "BATCHNORM op: the shapes of input arrays are not mutually broadcastable !");

    const bool applyScale  = (bool)INT_ARG(0);
    const bool applyOffset = (bool)INT_ARG(1);
    const T    epsilon     = T_ARG(0);

    // normalized output = gamma * ((input - mean) / sqrt(variance + epsilon)) + beta
    
    NDArray<T> inv = (*variance + epsilon).template transform<simdOps::RSqrt<T>>();
    if(applyScale)
        inv *= *gamma;
    
    if (applyOffset)
        *output = (*input) * inv - (*mean) * inv + *beta;
    else 
        *output = (*input) * inv - (*mean) * inv;

    STORE_RESULT(*output);
 
    return ND4J_STATUS_OK;
}


        // //////////////////////////////////////////////////////////////////////////
        // CONFIGURABLE_OP_IMPL(batchnorm_bp, 5, 1, true, 0, 1) {

        //     NDArray<T>* input = INPUT_VARIABLE(0);
        //     NDArray<T>* epsilon = INPUT_VARIABLE(1);
        //     NDArray<T>* gamma = INPUT_VARIABLE(2);
        //     NDArray<T>* dGlobalMeanView = INPUT_VARIABLE(3);
        //     NDArray<T>* dGlobalVarView = INPUT_VARIABLE(4);
        //     NDArray<T>* outEpsilon = this->getZ(block);
        //     std::vector<int> argI = *(block.getIArguments());
        //     const int bS = epsilon->sizeAt(0);
        //     bool isLockGammaBeta = (bool)argI[0];
        //     const int* epsilonShape = epsilon->getShapeInfo() + 1;
        //     const T eps = (T)1e-5;

        //     int rank = epsilon->rankOf();
        //     std::initializer_list<int> dimensions;
        //     int effectiveBatchSize;
        //     if (rank == 2) {
        //         dimensions = {0};
        //         effectiveBatchSize = bS;
        //     }
        //     else if (rank == 4) {
        //         dimensions = {0, 2, 3};
        //         effectiveBatchSize = input->sizeAt(0)*input->sizeAt(2)*input->sizeAt(3);
        //     }
        //     else
        //         throw "Graph operation batchnorm_bp: the epsilon rank must be equal to 2 or 4 !";

        //     NDArray<T> *mean(nullptr), *var(nullptr), *dBeta(nullptr), *dGamma(nullptr), *dLdVar(nullptr), *dxmu1(nullptr), *dxmu2(nullptr);
        //     mean = input->template reduceAlongDimension<simdOps::Mean<T>>(dimensions);
        //     var = input->template varianceAlongDimension<simdOps::SummaryStatsVariance<T>>(false, dimensions);
        //     var->template applyScalar<simdOps::Add<T>>(eps, nullptr);
        //     auto std = new NDArray<T>(var->getShapeInfo(), block.getWorkspace());
        //     var->template applyTransform<simdOps::Sqrt<T>>(std, nullptr);

        //     auto xMu = new NDArray<T>(input->getShapeInfo(), block.getWorkspace());
        //     auto xHat = new NDArray<T>(input->getShapeInfo(), block.getWorkspace());
        //     auto temp1 = new NDArray<T>(epsilon->getShapeInfo(), block.getWorkspace());
        //     auto temp2 = new NDArray<T>(std->getShapeInfo(), block.getWorkspace());
        //     auto dGammaView = new NDArray<T>('c', {1, epsilonShape[1]}, block.getWorkspace());
        //     auto dBetaView = new NDArray<T>('c', {1, epsilonShape[1]}, block.getWorkspace());
        //     auto dxhat = new NDArray<T>(epsilon->getShapeInfo(), block.getWorkspace());

        //     if (rank == 2) {
        //         input->subRowVector(mean, xMu);
        //         xMu->divRowVector(std, xHat);
        //     }
        //     else {
        //         input->template applyBroadcast<simdOps::Subtract<T>>({1}, mean, xMu, nullptr);
        //         xMu->template applyBroadcast<simdOps::Divide<T>>({1}, std, xHat, nullptr);
        //     }

        //     dBeta = epsilon->sum(dimensions); // dL/dBeta = sum_examples dL/dOut
        //     epsilon->template applyPairwiseTransform<simdOps::Multiply<T>>(xHat, temp1, nullptr);   //dL/dGamma = sum_examples dL/dOut .* xHat
        //     dGamma = temp1->sum(dimensions);  //dL/dGamma = sum_examples dL/dOut .* xHat

        //     if (isLockGammaBeta)
        //         epsilon->template applyPairwiseTransform<simdOps::Multiply<T>>(gamma, dxhat, nullptr);
        //     else {// Standard case
        //         if(rank == 2)
        //             epsilon->mulRowVector(gamma, dxhat); //dL/dxHat = dL/dOut . gamma        Shape: [minibatchSize, nOut]
        //         else
        //             epsilon->template applyBroadcast<simdOps::Multiply<T>>({1}, gamma, dxhat, nullptr);
        //     }

        //     // dLdVar - dL/dVariance, shape: [1, miniBatch]
        //     dxhat->template applyPairwiseTransform<simdOps::Multiply<T>>(xMu, temp1, nullptr);
        //     dLdVar = temp1->sum(dimensions);
        //     dLdVar->template applyScalar<simdOps::Multiply<T>>((T)-0.5, nullptr);
        //     T powParams[] = {(T)(-3.)};
        //     std->template applyTransform<simdOps::Pow<T>>(temp2, powParams);
        //     dLdVar->template applyPairwiseTransform<simdOps::Multiply<T>>(temp2, nullptr);

        //     //dL/dmu
        //     dxmu1 = dxhat->sum(dimensions);
        //     dxmu1->template applyPairwiseTransform<simdOps::Divide<T>>(std, nullptr);
        //     dxmu1->template applyTransform<simdOps::Neg<T>>();
        //     dxmu2 = xMu->sum(dimensions);
        //     dxmu2->template applyScalar<simdOps::Multiply<T>>((T)(-2.)/effectiveBatchSize);
        //     dxmu2->template applyPairwiseTransform<simdOps::Multiply<T>>(dLdVar, nullptr);

        //     dxmu1->template applyPairwiseTransform<simdOps::Add<T>>(dxmu2, nullptr);
        //     NDArray<T>* dLdmu = dxmu1;      //  = dL/dmu Shape: [1, nOut]

        //     //Note the array reuse here: dxhat, xMu, dLdVar, dLdmu - all are invalid after this line (but aren't used later anyway)
        //     NDArray<T>* dLdx = dxhat;
        //     dLdVar->template applyScalar<simdOps::Multiply<T>>((T)(2.)/effectiveBatchSize);
        //     dLdmu->template applyScalar<simdOps::Multiply<T>>((T)(1.)/effectiveBatchSize);
        //     if(rank == 2) {
        //         dLdx->divRowVector(std, dLdx);
        //         xMu->mulRowVector(dLdVar, xMu);
        //     }
        //     else {
        //         dLdx->template applyBroadcast<simdOps::Divide<T>>({1}, std, dLdx, nullptr);
        //         xMu->template applyBroadcast<simdOps::Multiply<T>>({1}, dLdVar, xMu, nullptr);
        //     }
        //     dLdx->template applyPairwiseTransform<simdOps::Add<T>>(xMu, nullptr);
        //     if(rank == 2)
        //         dLdx->addRowVector(dLdmu, dLdx);
        //     else
        //         dLdx->template applyBroadcast<simdOps::Add<T>>({1}, dLdmu, dLdx, nullptr);

        //     *outEpsilon = *dLdx;

        //     //TODO rework this to avoid the assign here
        //     // dGammaView->assign(dGamma);
        //     // dBetaView->assign(dBeta);
        //     // dGlobalMeanView->assign((T)0.);
        //     // dGlobalVarView->assign((T)0.);
        //     // retGradient.setGradientFor(BatchNormalizationParamInitializer.GAMMA, dGammaView);
        //     // retGradient.setGradientFor(BatchNormalizationParamInitializer.BETA, dBetaView);
        //     // retGradient.setGradientFor(BatchNormalizationParamInitializer.GLOBAL_MEAN, dGlobalMeanView);
        //     // retGradient.setGradientFor(BatchNormalizationParamInitializer.GLOBAL_VAR, dGlobalVarView);

        //     delete std;
        //     delete xMu;
        //     delete xHat;
        //     delete mean;
        //     delete var;
        //     delete dBeta;
        //     delete dGamma;
        //     delete dLdVar;
        //     delete dxmu1;
        //     delete dxmu2;
        //     delete temp1;
        //     delete temp2;
        //     delete dxhat;
        //     delete dGammaView;
        //     delete dBetaView;

        //     return ND4J_STATUS_OK;
        // }



DECLARE_SHAPE_FN(batchnorm) {
    
    Nd4jLong* outShapeInfo = nullptr;
    
    // evaluate output shapeInfo which is common broadcast shape for all input arrays
    ShapeUtils<T>::evalCommonBroadcastShapeInfo({INPUT_VARIABLE(0),INPUT_VARIABLE(1),INPUT_VARIABLE(2),INPUT_VARIABLE(3),INPUT_VARIABLE(4)}, outShapeInfo, block.getWorkspace());
 
    return SHAPELIST(outShapeInfo);
}





}
}

#endif