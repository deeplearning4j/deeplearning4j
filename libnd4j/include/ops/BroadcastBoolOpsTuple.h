/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author raver119@gmail.com
//

#ifndef DEV_TESTS_BROADCASTBOOLOPSTUPLE_H
#define DEV_TESTS_BROADCASTBOOLOPSTUPLE_H
#include <system/common.h>
#include <system/op_enums.h>

namespace sd {
class SD_LIB_EXPORT BroadcastBoolOpsTuple {
 private:
 public:
  scalar::BoolOps s;
  pairwise::BoolOps p;
  broadcast::BoolOps b;

  BroadcastBoolOpsTuple() = default;
  ~BroadcastBoolOpsTuple() = default;

  BroadcastBoolOpsTuple(scalar::BoolOps scalar, pairwise::BoolOps pairwise, broadcast::BoolOps broadcast) {
    s = scalar;
    p = pairwise;
    b = broadcast;
  }

  static BroadcastBoolOpsTuple custom(scalar::BoolOps scalar, pairwise::BoolOps pairwise, broadcast::BoolOps broadcast);
};
}  // namespace sd

#endif  // DEV_TESTS_BROADCASTOPSTUPLE_H
