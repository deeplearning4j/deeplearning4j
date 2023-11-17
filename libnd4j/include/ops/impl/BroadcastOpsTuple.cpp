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
#include <ops/BroadcastOpsTuple.h>

namespace sd {
BroadcastOpsTuple BroadcastOpsTuple::custom(scalar::Ops scalar, pairwise::Ops pairwise, broadcast::Ops broadcast) {
  BroadcastOpsTuple t(scalar, pairwise, broadcast);
  return t;
}

BroadcastOpsTuple BroadcastOpsTuple::Add() { return custom(scalar::Add, pairwise::Add, broadcast::Add); }

BroadcastOpsTuple BroadcastOpsTuple::Assign() {
  return custom(scalar::CopyPws, pairwise::CopyPws, broadcast::CopyPws);
}

BroadcastOpsTuple BroadcastOpsTuple::Divide() {
  return custom(scalar::Divide, pairwise::Divide, broadcast::Divide);
}

BroadcastOpsTuple BroadcastOpsTuple::DivideNoNan() {
  return custom(scalar::DivideNoNan, pairwise::DivideNoNan, broadcast::DivideNoNan);
}

BroadcastOpsTuple BroadcastOpsTuple::Multiply() {
  return custom(scalar::Multiply, pairwise::Multiply, broadcast::Multiply);
}

BroadcastOpsTuple BroadcastOpsTuple::Subtract() {
  return custom(scalar::Subtract, pairwise::Subtract, broadcast::Subtract);
}
BroadcastOpsTuple BroadcastOpsTuple::IGamma() {
  return custom(scalar::IGamma, pairwise::IGamma, broadcast::IGamma);
}
BroadcastOpsTuple BroadcastOpsTuple::IGammac() {
  return custom(scalar::IGammac, pairwise::IGammac, broadcast::IGammac);
}

BroadcastOpsTuple BroadcastOpsTuple::Pow() { return custom(scalar::Pow, pairwise::Pow, broadcast::Pow); }
BroadcastOpsTuple BroadcastOpsTuple::PowDerivative() {
  return custom(scalar::PowDerivative, pairwise::PowDerivative, broadcast::PowDerivative);
}

}  // namespace sd
