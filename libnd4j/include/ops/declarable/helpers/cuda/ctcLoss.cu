/*******************************************************************************
 * Copyright (c) 2021 Konduit K.K.
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
// @author AbdelRauf
//

#include <type_traits>
#include <cmath>
#include <stdexcept>
#include <memory>
#include <execution/Threads.h>
#include <execution/ThreadPool.h>
#include <helpers/LoopsCoordsHelper.h>
#include <ops/declarable/helpers/ctcLoss.h>

namespace sd
{
    namespace ops
    {
        namespace helpers
        {

 
           void ctcLoss(graph::Context& block, const NDArray &logInput, const NDArray &targetLabels, const NDArray &logInputLengths, const NDArray &targetLabelLengths, NDArray &logLosses, NDArray &gradients, int blankIndex){
                  //not imeplemented
                  throw std::runtime_error("ctcLoss:: Not implemented yet");
			}

 
        } // namespace helpers
    }     // namespace ops
} // namespace sd