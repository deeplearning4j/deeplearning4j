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
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/lrn.h>
#include <Status.h>

namespace nd4j {
namespace ops {
namespace helpers {

    // FIXME: double
    int lrnFunctor(NDArray* input, NDArray* output, int depth, double bias, double alpha, double beta) {

        double dividor;

        int totalLength = input->lengthOf();
        int lastDim = input->sizeAt(-1);
        int chunkCount = totalLength / lastDim;
        std::unique_ptr<ResultSet> listOut(output->allTensorsAlongDimension({output->rankOf() - 1}));
        std::unique_ptr<ResultSet> listInput(input->allTensorsAlongDimension({input->rankOf() - 1}));
        if (chunkCount != listOut->size()) 
            return ND4J_STATUS_VALIDATION;
        for (int c = 0; c < chunkCount; c++) {
            for (int e = 0; e < lastDim; e++) {
                int begin = nd4j::math::nd4j_max(0, e - depth);
                int end = nd4j::math::nd4j_min(depth + e + 1, lastDim);
                double quadSum = 0;

                for (int pos = begin; pos < end; ++pos) {
                    double val = listInput->at(c)->e<double>(pos);
                    quadSum += val * val;
                }
                double dividor = nd4j::math::nd4j_pow<double, double, double>(bias + alpha * quadSum, beta);
                listOut->at(c)->putScalar<double>(e,  listInput->at(c)->e<double>(e) / dividor);
            }
        }

        return Status::OK();
    }

    int lrnFunctorEx(NDArray* input, NDArray* output, NDArray* unitScale, NDArray* scale, int depth, double bias, double alpha, double beta) {
    
        depth = nd4j::math::nd4j_min<Nd4jLong>(depth, input->sizeAt(1));

        int halfDepth = depth / 2;
        halfDepth = nd4j::math::nd4j_max(halfDepth, 0);
        const int channel =  input->sizeAt(1);

        std::unique_ptr<NDArray> activitySqr(input->dup('c'));//NDArrayFactory<T>::createUninitialized(input));
        std::unique_ptr<NDArray> sumPart(activitySqr->dup('c'));

        input->applyPairwiseTransform(pairwise::Multiply, input, activitySqr.get(), nullptr);
#pragma omp parallel for if (halfDepth + 1 > Environment::getInstance()->elementwiseThreshold()) schedule(static)         
        for (int i = 1; i < halfDepth + 1; i++) {
            IndicesList indA({NDIndex::all(), NDIndex::interval(i, channel), NDIndex::all(), NDIndex::all()});
            IndicesList indB({NDIndex::all(), NDIndex::interval(0, channel - i), NDIndex::all(), NDIndex::all()});

            std::unique_ptr<NDArray> tmp(sumPart->subarray(indA));
            std::unique_ptr<NDArray> addVal(activitySqr->subarray(indB));

            tmp->applyPairwiseTransform(pairwise::Add, addVal.get(), nullptr);


            std::unique_ptr<NDArray> tmp2(sumPart->subarray(indB));
            std::unique_ptr<NDArray> addVal2(activitySqr->subarray(indA));

            tmp2->applyPairwiseTransform(pairwise::Add, addVal2.get(), nullptr);
        }

        /*
         *  // taken from java
            unitScale = sumPart.mul(alpha).addi(k).leverageTo(ComputationGraph.workspaceExternal);
            // y = x * unitScale**-beta
            scale = Transforms.pow(unitScale, -beta).leverageTo(ComputationGraph.workspaceExternal);
            activations = input.mul(scale).leverageTo(ComputationGraph.workspaceExternal);
         */
        if (unitScale != nullptr && scale != nullptr) {
            sumPart->applyScalar(scalar::Multiply, alpha, unitScale, nullptr);
            unitScale->applyScalar(scalar::Add, bias);

            float p = static_cast<float>(-beta);
            unitScale->applyScalar(scalar::Pow, p, scale, nullptr);
            input->applyPairwiseTransform(pairwise::Multiply, scale, output, nullptr);
        }

        return Status::OK();
    }

}
}
}
