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
#include <array/NDArrayFactory.h>
#include <helpers/ShapeUtils.h>
#include <system/op_boilerplate.h>


namespace sd {
namespace ops {
namespace helpers {

template <typename T>
void maximumBPFunctor_(NDArray* x, NDArray* y, NDArray* epsNext, NDArray* gradX, NDArray* gradY) {
  if (x->isSameShape(y)) {
    // Same-shape pairwise case.
    // gradX[i] = epsNext[i] where x[i] >= y[i], else 0  (ties go to both)
    // gradY[i] = epsNext[i] where x[i] <= y[i], else 0  (ties go to both)

    // getShapeAsVector() returns heap std::vector<LongType>* — store and delete.
    auto xShapeVec = x->getShapeAsVector();

    // X gradient: bool mask = (x >= y), cast to numeric type, multiply with epsNext
    auto maskBoolX = NDArrayFactory::create('c', *xShapeVec, DataType::BOOL, x->getContext());
    x->applyPairwiseTransform(pairwise::GreaterThanOrEqual, y, maskBoolX, nullptr);
    auto maskX = maskBoolX->cast(x->dataType());
    epsNext->applyPairwiseTransform(pairwise::Multiply, maskX, gradX, nullptr);
    delete maskX;
    delete maskBoolX;

    // Y gradient: bool mask = (x <= y), cast to numeric type, multiply with epsNext
    auto maskBoolY = NDArrayFactory::create('c', *xShapeVec, DataType::BOOL, x->getContext());
    x->applyPairwiseTransform(pairwise::LessThanOrEqual, y, maskBoolY, nullptr);
    auto maskY = maskBoolY->cast(x->dataType());
    epsNext->applyPairwiseTransform(pairwise::Multiply, maskY, gradY, nullptr);
    delete maskY;
    delete maskBoolY;

    delete xShapeVec;

  } else if (y->isScalar()) {
    // Scalar-y case.
    // gradX[i] = epsNext[i] where x[i] >= s, else 0
    // gradY   = sum of epsNext[i] where x[i] <= s
    T s = y->e<T>(0);

    auto xShapeVec = x->getShapeAsVector();

    // gradX: bool mask = (x >= s), cast, multiply
    auto maskBoolX = NDArrayFactory::create('c', *xShapeVec, DataType::BOOL, x->getContext());
    x->applyScalar<T>(scalar::GreaterThanOrEqual, s, maskBoolX, nullptr);
    auto maskX = maskBoolX->cast(x->dataType());
    epsNext->applyPairwiseTransform(pairwise::Multiply, maskX, gradX, nullptr);
    delete maskX;
    delete maskBoolX;

    // gradY: sum(epsNext * mask(x <= s)) — reduces to scalar
    auto maskBoolY = NDArrayFactory::create('c', *xShapeVec, DataType::BOOL, x->getContext());
    x->applyScalar<T>(scalar::LessThanOrEqual, s, maskBoolY, nullptr);
    auto maskY = maskBoolY->cast(x->dataType());
    auto weighted = NDArrayFactory::create('c', *xShapeVec, x->dataType(), x->getContext());
    epsNext->applyPairwiseTransform(pairwise::Multiply, maskY, weighted, nullptr);
    weighted->reduceNumber(reduce::Sum, gradY);
    delete weighted;
    delete maskY;
    delete maskBoolY;

    delete xShapeVec;

  } else {
    // Broadcast case.
    // Tile x and y to the shape of epsNext, compute masks separately into
    // separate output arrays (gradXFull, gradYFull) to avoid the data-dependency
    // bug where the original code overwrote preX then read it again.

    // getShapeAsVector() heap pointer — delete after use.
    auto targetShape = epsNext->getShapeAsVector();

    auto tiledX = x->dup();
    auto tiledY = y->dup();
    tiledX->tileToShape(*targetShape, *tiledX);
    tiledY->tileToShape(*targetShape, *tiledY);

    // gradXFull = epsNext * (tiledX >= tiledY)
    auto maskBoolX = NDArrayFactory::create('c', *targetShape, DataType::BOOL, x->getContext());
    tiledX->applyPairwiseTransform(pairwise::GreaterThanOrEqual, tiledY, maskBoolX, nullptr);
    auto maskX = maskBoolX->cast(x->dataType());
    auto gradXFull = NDArrayFactory::create('c', *targetShape, x->dataType(), x->getContext());
    epsNext->applyPairwiseTransform(pairwise::Multiply, maskX, gradXFull, nullptr);
    delete maskX;
    delete maskBoolX;

    // gradYFull = epsNext * (tiledX <= tiledY)
    auto maskBoolY = NDArrayFactory::create('c', *targetShape, DataType::BOOL, x->getContext());
    tiledX->applyPairwiseTransform(pairwise::LessThanOrEqual, tiledY, maskBoolY, nullptr);
    auto maskY = maskBoolY->cast(x->dataType());
    auto gradYFull = NDArrayFactory::create('c', *targetShape, x->dataType(), x->getContext());
    epsNext->applyPairwiseTransform(pairwise::Multiply, maskY, gradYFull, nullptr);
    delete maskY;
    delete maskBoolY;

    delete tiledX;
    delete tiledY;

    // Reduce back to original x/y shapes via sum along broadcast axes
    auto axisX = ShapeUtils::evalBroadcastBackwardAxis(x->shapeInfo(), epsNext->shapeInfo());
    auto axisY = ShapeUtils::evalBroadcastBackwardAxis(y->shapeInfo(), epsNext->shapeInfo());

    if (axisX.size() > 0) {
      auto sum = gradXFull->reduceAlongDimension(reduce::Sum, &axisX);
      gradX->assign(sum);
      delete sum;
    } else {
      gradX->assign(gradXFull);
    }

    if (axisY.size() > 0) {
      auto sum = gradYFull->reduceAlongDimension(reduce::Sum, &axisY);
      gradY->assign(sum);
      delete sum;
    } else {
      gradY->assign(gradYFull);
    }

    delete gradXFull;
    delete gradYFull;
    delete targetShape;
  }
}
BUILD_SINGLE_TEMPLATE(void maximumBPFunctor_, (NDArray* x, NDArray* y, NDArray* epsNext, NDArray* gradX, NDArray* gradY), SD_NUMERIC_TYPES);

void maximumBPFunctor(LaunchContext* context, NDArray* x, NDArray* y, NDArray* epsNext, NDArray* gradX,
                      NDArray* gradY) {
  NDArray::prepareSpecialUse({gradX, gradY}, {x, y, epsNext});

  BUILD_SINGLE_SELECTOR(x->dataType(), maximumBPFunctor_, (x, y, epsNext, gradX, gradY), SD_NUMERIC_TYPES);

  NDArray::registerSpecialUse({gradX, gradY}, {x, y, epsNext});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
