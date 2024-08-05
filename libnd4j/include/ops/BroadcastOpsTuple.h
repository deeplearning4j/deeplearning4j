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

#ifndef DEV_TESTS_BROADCASTOPSTUPLE_H
#define DEV_TESTS_BROADCASTOPSTUPLE_H
#include <system/common.h>
#include <system/op_enums.h>

namespace sd {
class SD_LIB_EXPORT BroadcastOpsTuple {
 private:
 public:
  scalar::Ops s;
  pairwise::Ops p;
  broadcast::Ops b;

  BroadcastOpsTuple() = default;
  ~BroadcastOpsTuple() = default;

  BroadcastOpsTuple(scalar::Ops scalar, pairwise::Ops pairwise, broadcast::Ops broadcast) {
    s = scalar;
    p = pairwise;
    b = broadcast;
  }

  static BroadcastOpsTuple custom(scalar::Ops scalar, pairwise::Ops pairwise, broadcast::Ops broadcast);

  static BroadcastOpsTuple Add();
  static BroadcastOpsTuple Assign();
  static BroadcastOpsTuple Divide();
  static BroadcastOpsTuple DivideNoNan();
  static BroadcastOpsTuple Multiply();
  static BroadcastOpsTuple Subtract();
  static BroadcastOpsTuple IGamma();
  static BroadcastOpsTuple IGammac();

  static BroadcastOpsTuple Pow();
  static BroadcastOpsTuple PowDerivative();
};
}  // namespace sd

#endif  // DEV_TESTS_BROADCASTOPSTUPLE_H
