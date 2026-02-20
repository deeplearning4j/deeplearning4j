/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#include <ops/declarable/helpers/ema_update.h>

namespace sd {
namespace ops {
namespace helpers {

void emaUpdate(NDArray* model, NDArray* shadow, NDArray* output,
                  double decay, LaunchContext* context) {
    // EMA forward: output = decay * shadow + (1 - decay) * model
    auto temp1 = (*shadow) * decay;
    auto temp2 = (*model) * (1.0 - decay);
    output->assign(temp1 + temp2);
}

void emaUpdateBp(NDArray* model, NDArray* shadow, NDArray* gradOutput,
                    NDArray* dLdModel, NDArray* dLdShadow,
                    double decay, LaunchContext* context) {
    // d(output)/d(model) = (1 - decay)
    // d(output)/d(shadow) = decay
    dLdModel->assign((*gradOutput) * (1.0 - decay));
    dLdShadow->assign((*gradOutput) * decay);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
