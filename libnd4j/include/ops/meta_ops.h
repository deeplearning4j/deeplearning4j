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

#pragma once
#ifndef FUSED_OPS_H_
#define FUSED_OPS_H_

#include <ops/ops.h>
#include <system/op_boilerplate.h>

namespace metaOps {
/**
 * InvertedMetaOp shares the same idea as MetaOp, but op being applied to op.Y in pairwise/broadcast ops
 */
template <typename T, typename OpTypeA, typename OpTypeB>
class InvertedMetaOp {
 public:
  no_op_exec_special no_op_exec_special_cuda

      /*
       * PREDICATE
       */

      // scalar, transform, reduce, indexreduce entry
      SD_OP_DEF static T
      op(T d1, T *params) {
    /*
     * We assume, that this method won't be EVER called
     */
    printf("You should NEVER see this message in output\n");
    return (T)0.0f;
  }

  // PWT, broadcast entry. Predicate can be only scalar, transform
  SD_OP_DEF static T op(T d1, T d2, T *params) {
    sd::Pointer *wrap = reinterpret_cast<sd::Pointer *>(params);
    T *paramsA = reinterpret_cast<T *>(wrap[0]);
    T *paramsB = reinterpret_cast<T *>(wrap[1]);

    return OpTypeB::op(OpTypeA::op(d1, d2, paramsA), paramsB);
  }

  /*
   * POSTULATE
   */

  // will be called for reduce, reduce3
  SD_OP_DEF static T postProcess(T reduction, sd::LongType n, T *params) {
    /*
     * We assume, that this method won't be EVER called
     */
    printf("You should NEVER EVER see this message in output\n");

    return (T)0.0f;
  }
};

/**
 * Special case here: MetaOp which consist of 2 operations.
 *
 * Predicate can be either scalar or transform, to process data before actual op call
 * Postulate will be the scalar/transform, but will be applied to result of broadcast/reduce/reduce3
 */
template <typename T, typename OpTypeA, typename OpTypeB>
class MetaOp {
 public:
  no_op_exec_special no_op_exec_special_cuda

      /*
       * PREDICATE
       */

      SD_META_DEF static T
      startingValue(const T *input) {
    return (T)0.0f;
  }

  // scalar, transform, reduce, indexreduce entry
  SD_META_DEF static T op(T d1, T *params) {
    /*
     * We assume, that params for MetaOp is a set of pointers to actual op A & B extraArgs
     */
    sd::Pointer *wrap = reinterpret_cast<sd::Pointer *>(params);
    T *paramsA = reinterpret_cast<T *>(wrap[0]);
    T *paramsB = reinterpret_cast<T *>(wrap[1]);

    return OpTypeB::op(OpTypeA::op(d1, paramsA), paramsB);
  }

  // PWT, broadcast entry. Predicate can be only scalar, transform
  SD_META_DEF static T op(T d1, T d2, T *params) {
    sd::Pointer *wrap = reinterpret_cast<sd::Pointer *>(params);
    T *paramsA = reinterpret_cast<T *>(wrap[0]);
    T *paramsB = reinterpret_cast<T *>(wrap[1]);

    return OpTypeB::op(OpTypeA::op(d1, paramsA), d2, paramsB);
  }

  /*
   * POSTULATE
   */

  // will be called for reduce, reduce3
  SD_META_DEF static T postProcess(T reduction, sd::LongType n, T *params) {
    sd::Pointer *wrap = reinterpret_cast<sd::Pointer *>(params);
    T *paramsA = reinterpret_cast<T *>(wrap[0]);
    T *paramsB = reinterpret_cast<T *>(wrap[1]);

    return OpTypeB::op(OpTypeA::postProcess(reduction, n, paramsA), paramsB);
  }
};

template <typename T, typename OpTypeA, typename OpTypeB>
class ReduceMetaOp {
 public:
  no_op_exec_special no_op_exec_special_cuda

      SD_META_DEF static T
      startingValue(const T *input) {
    return OpTypeB::startingValue(input);
  }

  SD_META_DEF static T merge(T old, T opOutput, T *params) {
    sd::Pointer *wrap = reinterpret_cast<sd::Pointer *>(params);
    //            T *paramsA = reinterpret_cast<T *> (wrap[0]);
    T *paramsB = reinterpret_cast<T *>(wrap[1]);

    return OpTypeB::merge(old, opOutput, paramsB);
  }

  SD_META_DEF static T update(T old, T opOutput, T *params) {
    sd::Pointer *wrap = reinterpret_cast<sd::Pointer *>(params);
    // T *paramsA = reinterpret_cast<T *> (wrap[0]);
    T *paramsB = reinterpret_cast<T *>(wrap[1]);

    return OpTypeB::update(old, opOutput, paramsB);
  }

  SD_META_DEF static T op(T d1, T *params) {
    sd::Pointer *wrap = reinterpret_cast<sd::Pointer *>(params);
    T *paramsA = reinterpret_cast<T *>(wrap[0]);
    T *paramsB = reinterpret_cast<T *>(wrap[1]);

    return OpTypeB::op(OpTypeA::op(d1, paramsA), paramsB);
  }

  SD_META_DEF static T postProcess(T reduction, sd::LongType n, T *params) {
    sd::Pointer *wrap = reinterpret_cast<sd::Pointer *>(params);
    //            T *paramsA = reinterpret_cast<T *> (wrap[0]);
    T *paramsB = reinterpret_cast<T *>(wrap[1]);

    return OpTypeB::postProcess(reduction, n, paramsB);
  }
};
}  // namespace metaOps

#endif
