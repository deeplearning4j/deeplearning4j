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
// Created by raver119 on 07.10.2017.
//

#ifndef LIBND4J_CUSTOMOPERATIONS_H
#define LIBND4J_CUSTOMOPERATIONS_H
#include <array/NDArrayFactory.h>
#include <helpers/ArrayUtils.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/OpTracker.h>
#include <helpers/ShapeBuilders.h>
#include <helpers/TAD.h>
#include <helpers/shape.h>
#include <ops/declarable/headers/BarnesHutTsne.h>
#include <ops/declarable/headers/activations.h>
#include <ops/declarable/headers/bitwise.h>
#include <ops/declarable/headers/blas.h>
#include <ops/declarable/headers/boolean.h>
#include <ops/declarable/headers/broadcastable.h>
#include <ops/declarable/headers/compat.h>
#include <ops/declarable/headers/compression.h>
#include <ops/declarable/headers/convo.h>
#include <ops/declarable/headers/datatypes.h>
#include <ops/declarable/headers/decoder.h>
#include <ops/declarable/headers/images.h>
#include <ops/declarable/headers/kernels.h>
#include <ops/declarable/headers/list.h>
#include <ops/declarable/headers/loss.h>
#include <ops/declarable/headers/nlp.h>
#include <ops/declarable/headers/nn.h>
#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/headers/random.h>
#include <ops/declarable/headers/recurrent.h>
#include <ops/declarable/headers/shape.h>
#include <ops/declarable/headers/strings.h>
#include <ops/declarable/headers/tests.h>
#include <ops/declarable/headers/third_party.h>
#include <ops/declarable/headers/transforms.h>
#include <ops/declarable/headers/updaters.h>
#include <ops/declarable/headers/util.h>

namespace sd {
     class SD_LIB_EXPORT  _loader {
     public:
         _loader();
};

namespace ops {

// logic ops
#if NOT_EXCLUDED(OP_Switch)
DECLARE_DIVERGENT_OP(Switch, 2, 2, true);
#endif
#if NOT_EXCLUDED(OP_While)
DECLARE_LOGIC_OP(While);
#endif
#if NOT_EXCLUDED(OP_Scope)
DECLARE_LOGIC_OP(Scope);
#endif
#if NOT_EXCLUDED(OP_Conditional)
DECLARE_LOGIC_OP(Conditional);
#endif
#if NOT_EXCLUDED(OP_Return)
DECLARE_LOGIC_OP(Return);
#endif

/**
 * This operations exposes given arguments as it's own outputs, but does it only once.
 * Subsequent calls will be served directly by this op.
 *
 * PLEASE NOTE: This operation is internal graph operation, and shouldn't be used directly usually.
 */
#if NOT_EXCLUDED(OP_expose)
DECLARE_CUSTOM_OP(expose, -1, -1, true, 0, 0);
#endif
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_CUSTOMOPERATIONS_H
