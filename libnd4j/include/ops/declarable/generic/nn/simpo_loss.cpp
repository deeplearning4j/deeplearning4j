/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// Simple Preference Optimization (SimPO) loss op (arXiv:2405.14734)
//
// Forward:
//   diff_i  = beta * (chosen_lp_i - rejected_lp_i - gamma)
//   loss    = mean_i(-log(sigmoid(diff_i)))
//
// Backward (d/dx[-log(sigmoid(x))] = sigmoid(-x) = 1 - sigmoid(x)):
//   d_chosen_i   =  grad * (1 - sigmoid(diff_i)) * beta / N
//   d_rejected_i = -grad * (1 - sigmoid(diff_i)) * beta / N
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_simpo_loss)

#include <ops/declarable/headers/llm.h>
#include <math/templatemath.h>
#include <system/openmp_pragmas.h>

namespace sd {
namespace ops {

// ---------------------------------------------------------------------------
// Forward: simpo_loss
// ---------------------------------------------------------------------------
CUSTOM_OP_IMPL(simpo_loss, 2, 1, false, 2, 0) {
    auto chosenLP   = INPUT_VARIABLE(0);   // [batch]
    auto rejectedLP = INPUT_VARIABLE(1);   // [batch]
    auto output     = OUTPUT_VARIABLE(0);  // scalar

    const double beta  = T_ARG(0);
    const double gamma = T_ARG(1);

    const sd::LongType batchSize = chosenLP->lengthOf();

    double lossSum = 0.0;

    PRAGMA_OMP_PARALLEL_FOR_REDUCTION(+:lossSum)
    for (sd::LongType i = 0; i < batchSize; i++) {
        double diff = beta * (chosenLP->e<double>(i) - rejectedLP->e<double>(i) - gamma);
        // -log(sigmoid(diff)) = log(1 + exp(-diff))  (numerically stable via softplus)
        double val;
        if (diff >= 0.0) {
            val = sd::math::sd_log<double, double>(1.0 + sd::math::sd_exp<double, double>(-diff));
        } else {
            val = -diff + sd::math::sd_log<double, double>(1.0 + sd::math::sd_exp<double, double>(diff));
        }
        lossSum += val;
    }

    output->p(0, lossSum / static_cast<double>(batchSize));

    return sd::Status::OK;
}

DECLARE_TYPES(simpo_loss) {
  getOpDescriptor()->addTraits(OP_TRAIT_REDUCTION | OP_TRAIT_FULLY_WRITING);
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_FLOATS})
        ->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(simpo_loss) {
    // Output is a scalar
    auto outShape = ConstantShapeHelper::getInstance().scalarShapeInfo(
        ArrayOptions::dataType(inputShape->at(0)));
    return SHAPELIST(outShape);
}

// ---------------------------------------------------------------------------
// Backward: simpo_loss_bp
// ---------------------------------------------------------------------------
CUSTOM_OP_IMPL(simpo_loss_bp, 3, 2, false, 2, 0) {
    auto chosenLP   = INPUT_VARIABLE(0);   // [batch]
    auto rejectedLP = INPUT_VARIABLE(1);   // [batch]
    auto gradOut    = INPUT_VARIABLE(2);   // scalar upstream gradient
    auto dChosenLP  = OUTPUT_VARIABLE(0);  // [batch]
    auto dRejectedLP = OUTPUT_VARIABLE(1); // [batch]

    const double beta  = T_ARG(0);
    const double gamma = T_ARG(1);

    const sd::LongType batchSize = chosenLP->lengthOf();
    const double gradScalar = gradOut->e<double>(0);
    // Chain rule: mean reduces by 1/N, beta multiplies the inner gradient
    const double scale = gradScalar * beta / static_cast<double>(batchSize);

    PRAGMA_OMP_PARALLEL_FOR
    for (sd::LongType i = 0; i < batchSize; i++) {
        double diff = beta * (chosenLP->e<double>(i) - rejectedLP->e<double>(i) - gamma);
        // d/dx[-log(sigmoid(x))] = 1 - sigmoid(x) = sigmoid(-x)
        double sigmNeg;
        if (diff >= 0.0) {
            double expNeg = sd::math::sd_exp<double, double>(-diff);
            sigmNeg = expNeg / (1.0 + expNeg);
        } else {
            double expPos = sd::math::sd_exp<double, double>(diff);
            sigmNeg = 1.0 / (1.0 + expPos);
        }
        // d_chosen    = +scale * sigmNeg
        // d_rejected  = -scale * sigmNeg
        dChosenLP->p(i,  scale * sigmNeg);
        dRejectedLP->p(i, -scale * sigmNeg);
    }

    return sd::Status::OK;
}

DECLARE_TYPES(simpo_loss_bp) {
  getOpDescriptor()->addTraits(OP_TRAIT_REDUCTION | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD);
    getOpDescriptor()
        ->setAllowedInputTypes({ALL_FLOATS})
        ->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(simpo_loss_bp) {
    // Outputs mirror the shapes of chosen_lp and rejected_lp inputs
    auto chosen   = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(inputShape->at(0)),
        shape::order(inputShape->at(0)),
        shape::rank(inputShape->at(0)),
        shape::shapeOf(inputShape->at(0)));
    auto rejected = ConstantShapeHelper::getInstance().createShapeInfo(
        ArrayOptions::dataType(inputShape->at(1)),
        shape::order(inputShape->at(1)),
        shape::rank(inputShape->at(1)),
        shape::shapeOf(inputShape->at(1)));
    return SHAPELIST(chosen, rejected);
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_simpo_loss)
