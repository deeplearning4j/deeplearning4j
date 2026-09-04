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
//  @author raver119@gmail.com
//

#ifndef LIBND4J_BROADCAST_HELPER_H
#define LIBND4J_BROADCAST_HELPER_H

#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <helpers/ShapeUtils.h>
#include <ops/BroadcastBoolOpsTuple.h>
#include <ops/BroadcastOpsTuple.h>

namespace sd {
namespace ops {
class BroadcastHelper {
 public:
  static SD_INLINE NDArray* broadcastApply(BroadcastOpsTuple op, NDArray* x, NDArray* y, NDArray* z,
                                           ExtraArguments* extraArgs = nullptr) {
    if (x->isEmpty() || y->isEmpty()) {
      return z;
    }

    // Cache length values to avoid repeated virtual calls
    const sd::LongType xLen = x->lengthOf();
    const sd::LongType yLen = y->lengthOf();

    if (xLen > 1 && yLen > 1 && x->isSameShape(y)) {
      x->applyPairwiseTransform(op.p, y, z, extraArgs);
    } else if (xLen > 1 && yLen <= 1) {
      x->applyScalarArr(op.s, y, z);
    } else if (xLen <= 1 && yLen > 1) {
      if (z->isSameShape(y)) {
        if (op.s == scalar::Add || op.s == scalar::Multiply) {
          y->applyScalarArr(op.s, x, z);
        } else if (op.s == scalar::SquaredSubtract) {
          y->applyScalarArr(scalar::SquaredReverseSubtract, x, z);
        } else if (op.s == scalar::Subtract) {
          y->applyScalarArr(scalar::ReverseSubtract, x, z);
        } else if (op.s == scalar::Divide) {
          y->applyScalarArr(scalar::ReverseDivide, x, z);
        } else if (op.s == scalar::Pow) {
          y->applyScalarArr(scalar::ReversePow, x, z);
        } else if (op.s == scalar::ReverseSubtract) {
          y->applyScalarArr(scalar::Subtract, x, z);
        } else if (op.s == scalar::ReverseDivide) {
          y->applyScalarArr(scalar::Divide, x, z);
        } else if (op.s == scalar::MaxPairwise || op.s == scalar::MinPairwise || op.s == scalar::AMaxPairwise ||
                   op.s == scalar::AMinPairwise) {
          y->applyScalarArr(op.s, x, z);
        } else if (op.s == scalar::CopyPws) {
          z->assign(y);
        } else {
          z->assign(x);
          z->applyPairwiseTransform(op.p, y, extraArgs);
        }
        return z;
      } else {
        auto* yShapeVec = y->getShapeAsVector();
        auto tZ = NDArrayFactory::valueOf(*yShapeVec, y, y->ordering());
        delete yShapeVec;
        tZ->applyPairwiseTransform(op.p, y, extraArgs);
        return tZ;
      }
    } else if (xLen <= 1 && yLen <= 1) {
      x->applyScalarArr(op.s, y, z);
    } else if (ShapeUtils::areShapesBroadcastable(*x, *y)) {
      x->applyTrueBroadcast(op, y, z, true, extraArgs);
      return z;
    } else {
      auto sx = ShapeUtils::shapeAsString(x);
      auto sy = ShapeUtils::shapeAsString(y);
      const std::string message =
          "BroadcastHelper::broadcastApply cannot broadcast numeric inputs: "
          "x shape=" + sx + " dtype=" +
          std::to_string(static_cast<int>(x->dataType())) + ", y shape=" + sy +
          " dtype=" + std::to_string(static_cast<int>(y->dataType())) +
          ", output shape=" + ShapeUtils::shapeAsString(z);
#ifndef __JAVACPP_HACK__
      safeSetErrorContext(static_cast<int>(Status::KERNEL_FAILURE), message.c_str());
#endif
      return nullptr;
    }

    return z;
  }

  static SD_INLINE NDArray* broadcastApply(BroadcastBoolOpsTuple op, NDArray* x, NDArray* y, NDArray* z,
                                           ExtraArguments* extraArgs = nullptr) {
    if (x->isEmpty() || y->isEmpty()) {
      if (!z->isEmpty()) {
        std::string errorMessage;
        errorMessage += "BroadcastHelper::broadcastApply: when some of input arrays (or both) is empty, output array must be empty as well !";
        errorMessage += "X is empty: ";
        errorMessage += std::to_string(x->isEmpty());
        errorMessage += "Y is empty: ";
        errorMessage += std::to_string(y->isEmpty());
        THROW_EXCEPTION(errorMessage.c_str());
      }
      return z;
    }

    // Cache scalar checks to avoid repeated calls
    const bool xIsScalar = x->isScalar();
    const bool yIsScalar = y->isScalar();

    if (!xIsScalar && !yIsScalar && x->isSameShape(y)) {
      x->applyPairwiseTransform(op.p, y, z);
    } else if (!xIsScalar && yIsScalar) {
      x->applyScalarArr(op.s, y, z);
    } else if (xIsScalar && !yIsScalar) {
      // Scalar x broadcast against array y: use applyTrueBroadcast so every
      // element of y is compared against x.  applyPairwiseTransform must NOT
      // be used here because it iterates min(x.length, y.length) = 1 element,
      // leaving z[1..n-1] untouched (= false for bool output).
      x->applyTrueBroadcast(op, y, z, true, extraArgs);
      return z;
    } else if (xIsScalar && yIsScalar) {
      x->applyScalarArr(op.s, y, z);
    } else if (ShapeUtils::areShapesBroadcastable(*x, *y)) {
      x->applyTrueBroadcast(op, y, z, true, extraArgs);
      return z;
    } else {
      const std::string message =
          "BroadcastHelper::broadcastApply cannot broadcast boolean inputs: "
          "x shape=" + ShapeUtils::shapeAsString(x) + " dtype=" +
          std::to_string(static_cast<int>(x->dataType())) + ", y shape=" +
          ShapeUtils::shapeAsString(y) + " dtype=" +
          std::to_string(static_cast<int>(y->dataType())) +
          ", output shape=" + ShapeUtils::shapeAsString(z);
#ifndef __JAVACPP_HACK__
      safeSetErrorContext(static_cast<int>(Status::KERNEL_FAILURE), message.c_str());
#endif
      return nullptr;
    }

    return z;
  }
};
}  // namespace ops
}  // namespace sd

#endif
