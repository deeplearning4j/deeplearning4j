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
// @author Oleh Semeniv (oleg.semeniv@gmail.com)
//

#ifndef LIBND4J_UPDATER_RMS_PROM_H
#define LIBND4J_UPDATER_RMS_PROM_H
#include <array/NDArray.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

SD_LIB_HIDDEN void updaterRmsProp(sd::LaunchContext* context, const NDArray& gradient, const NDArray& initState,
                                  NDArray& update, NDArray& stateG, const double dLr, const double dRmsDecay,
                                  const double dEpsilon);
SD_LIB_HIDDEN void updaterAdaGrad(sd::LaunchContext* context, const NDArray& gradient, const NDArray& initState,
                                  NDArray& update, NDArray& stateH, const double dLr, const double dEpsilon);
SD_LIB_HIDDEN void updaterNesterovs(sd::LaunchContext* context, const NDArray& gradient, const NDArray& initState,
                                    NDArray& update, NDArray& stateV, const double dLr, const double bMomentum);
SD_LIB_HIDDEN void updaterAdaMax(sd::LaunchContext* context, const NDArray& gradient, const NDArray& initStateU,
                                 const NDArray& initStateM, NDArray& update, NDArray& stateU, NDArray& stateM,
                                 const double dLr, const double dBeta1, const double dBeta2, const double dEpsilon,
                                 const int nIteration);
SD_LIB_HIDDEN void updaterAdam(sd::LaunchContext* context, const NDArray& gradient, const NDArray& initStateU,
                               const NDArray& initStateM, NDArray& update, NDArray& stateU, NDArray& stateM,
                               const double dLr, const double dBeta1, const double dBeta2, const double dEpsilon,
                               const int nIteration);
SD_LIB_HIDDEN void updaterAdaDelta(sd::LaunchContext* context, const NDArray& gradient, const NDArray& initStateMsg,
                                   const NDArray& initStateMsdx, NDArray& update, NDArray& stateMsg, NDArray& stateMsdx,
                                   const double dRho, const double dEpsilon);
SD_LIB_HIDDEN void updaterNadam(sd::LaunchContext* context, const NDArray& gradient, const NDArray& initStateV,
                                const NDArray& initStateM, NDArray& update, NDArray& stateV, NDArray& stateM,
                                const double dLr, const double dBeta1, const double dBeta2, const double dEpsilon,
                                const int nIteration);
SD_LIB_HIDDEN void updaterAmsGrad(sd::LaunchContext* context, const NDArray& gradient, const NDArray& initStateV,
                                  const NDArray& initStateM, const NDArray& initStateH, NDArray& update,
                                  NDArray& stateV, NDArray& stateM, NDArray& stateH, const double dLr,
                                  const double dBeta1, const double dBeta2, const double dEpsilon,
                                  const int nIteration);
SD_LIB_HIDDEN void updaterAdaBelief(sd::LaunchContext* context, const NDArray& gradient, const NDArray& initStateU,
                                    const NDArray& initStateM, NDArray& update, NDArray& stateU, NDArray& stateM,
                                    const double dLr, const double dBeta1, const double dBeta2, const double dEpsilon,
                                    const int nIteration);
}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
