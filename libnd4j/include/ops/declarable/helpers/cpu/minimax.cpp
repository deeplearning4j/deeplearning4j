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
//  @author sgazeos@gmail.com
//
#include <array/NDArray.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/helpers/minimax.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void minimumBPFunctor_(LaunchContext* context, NDArray* x, NDArray* y, NDArray* epsNext, NDArray* gradX, NDArray* gradY) {
  auto lambdaX = LAMBDA_TTT(_e, _x, _y) { return _x <= _y ? _e : (T)0.; });

  auto lambdaY = LAMBDA_TTT(_e, _x, _y) { return _x >= _y ? _e : (T)0.; });

  if (x->isSameShape(y)) {
    // PWT case case

    // X gradient
    epsNext->applyTriplewiseLambda<T>(x, y, lambdaX, gradX);

    // Y gradient
    epsNext->applyTriplewiseLambda<T>(x, y, lambdaY, gradY);

  } else if (y->isScalar()) {
    T s = y->e<T>(0);
    auto lambdaS = LAMBDA_TT(_e, _x, s) { return _x <= s ? _e : (T)0.; });
    float zero = 0.0f;
    // scalar case
    auto tmp = epsNext->reduceNumber(reduce::Sum);
    if (x <= y)
      gradY->assign(&tmp);
    else
      gradY->assign(zero);

    epsNext->applyPairwiseLambda<T>(x, lambdaS, gradX);
  } else {
    // broadcast case

    // in this case we want to boost our X and Y shapes to the size of FF pass output (or epsNext, which has the same
    // shape)
    auto preX = x->dup();
    auto preY = y->dup();

    auto targetShape = epsNext->getShapeAsVector();

    preX.tileToShape(targetShape, preX);
    preY.tileToShape(targetShape, preY);

    epsNext->applyTriplewiseLambda<T>(&preX, &preY, lambdaX, &preX);
    epsNext->applyTriplewiseLambda<T>(&preX, &preY, lambdaY, &preY);

    auto axisX = ShapeUtils::evalBroadcastBackwardAxis(x->shapeInfo(), epsNext->shapeInfo());
    auto axisY = ShapeUtils::evalBroadcastBackwardAxis(y->shapeInfo(), epsNext->shapeInfo());

    if (axisX.size() > 0) {
      auto sum = preX.reduceAlongDimension(reduce::Sum, &axisX);
      gradX->assign(&sum);
    } else
      gradX->assign(&preX);

    if (axisY.size() > 0) {
      auto sum = preY.reduceAlongDimension(reduce::Sum, &axisY);
      gradY->assign(&sum);
    } else
      gradY->assign(&preY);
  }
}

template <typename T>
void maximumBPFunctor_(LaunchContext* context, NDArray* x, NDArray* y, NDArray* epsNext, NDArray* gradX, NDArray* gradY) {
  auto lambdaX = LAMBDA_TTT(_e, _x, _y) { return _x >= _y ? _e : (T)0.; });

  auto lambdaY = LAMBDA_TTT(_e, _x, _y) { return _x <= _y ? _e : (T)0.; });

  if (x->isSameShape(y)) {
    // PWT case case

    // X gradient
    epsNext->applyTriplewiseLambda<T>(x, y, lambdaX, gradX);

    // Y gradient
    epsNext->applyTriplewiseLambda<T>(x, y, lambdaY, gradY);

  } else if (y->isScalar()) {
    T s = y->e<T>(0);
    auto lambdaS = LAMBDA_TT(_e, _x, s) { return _x >= s ? _e : (T)0.; });

    // scalar case
    auto tmp = epsNext->reduceNumber(reduce::Sum);
    float zero = 0.0f;
    if (x <= y)
      gradY->assign(&tmp);
    else
      gradY->assign(zero);

    epsNext->applyPairwiseLambda<T>(x, lambdaS, gradX);
  } else {
    // broadcast case

    // in this case we want to boost our X and Y shapes to the size of FF pass output (or epsNext, which has the same
    // shape)
    auto preX = x->dup();
    auto preY = y->dup();

    auto targetShape = epsNext->getShapeAsVector();

    preX.tileToShape(targetShape, preX);
    preY.tileToShape(targetShape, preY);

    epsNext->applyTriplewiseLambda<T>(&preX, &preY, lambdaX, &preX);
    epsNext->applyTriplewiseLambda<T>(&preX, &preY, lambdaY, &preY);

    auto axisX = ShapeUtils::evalBroadcastBackwardAxis(x->shapeInfo(), epsNext->shapeInfo());
    auto axisY = ShapeUtils::evalBroadcastBackwardAxis(y->shapeInfo(), epsNext->shapeInfo());

    if (axisX.size() > 0) {
      auto sum = preX.reduceAlongDimension(reduce::Sum, &axisX);
      gradX->assign(&sum);
    } else
      gradX->assign(&preX);

    if (axisY.size() > 0) {
      auto sum = preY.reduceAlongDimension(reduce::Sum, &axisY);
      gradY->assign(&sum);
    } else
      gradY->assign(&preY);
  }
}

void minimumBPFunctor(LaunchContext* context, NDArray* x, NDArray* y, NDArray* epsNext, NDArray* gradX,
                      NDArray* gradY) {
  BUILD_SINGLE_SELECTOR(x->dataType(), minimumBPFunctor_, (context, x, y, epsNext, gradX, gradY), SD_NUMERIC_TYPES);
}

void maximumBPFunctor(LaunchContext* context, NDArray* x, NDArray* y, NDArray* epsNext, NDArray* gradX,
                      NDArray* gradY) {
  BUILD_SINGLE_SELECTOR(x->dataType(), maximumBPFunctor_, (context, x, y, epsNext, gradX, gradY), SD_NUMERIC_TYPES);
}
BUILD_SINGLE_TEMPLATE(template void minimumBPFunctor_,
                      (LaunchContext* context, NDArray* x, NDArray* y, NDArray* epsNext, NDArray* gradX, NDArray* gradY), SD_NUMERIC_TYPES);
BUILD_SINGLE_TEMPLATE(template void maximumBPFunctor_,
                      (LaunchContext* context, NDArray* x, NDArray* y, NDArray* epsNext, NDArray* gradX, NDArray* gradY), SD_NUMERIC_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd