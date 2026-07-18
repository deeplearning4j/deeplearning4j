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
// Created by raver119 on 16.10.2017.
//

#ifndef LIBND4J_LEGACYSCALAR_BOOL_OP_H
#define LIBND4J_LEGACYSCALAR_BOOL_OP_H
#include <ops/declarable/LegacyOp.h>

namespace sd {
namespace ops {
SD_BACKEND_OPS_INLINE_NAMESPACE_BEGIN
/**
 *   This class provides wrapper for scalar transform operations, i.e. a + b = c, where either a or b is scalar
 * primitive and other operand is NDArray
 */
class SD_LIB_EXPORT LegacyScalarBoolOp : public LegacyOp {
 protected:
  bool _cachedScalarValid = false;
  double _cachedScalarValue = 0.0;
  DataType _cachedScalarType = DataType::UNKNOWN;

  Status validateAndExecute(sd::graph::Context& block) override;

 public:
  LegacyScalarBoolOp();
  LegacyScalarBoolOp(int opNum);
  LegacyScalarBoolOp(int opNum, NDArray& scalar);

  ShapeList* calculateOutputShape(ShapeList* inputShape, Context& block) override;
  LegacyOp* clone() override;
  void registerTypes() override;
};
SD_BACKEND_OPS_INLINE_NAMESPACE_END
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_LEGACYSCALAROP_H
