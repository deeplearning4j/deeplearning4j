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
// @author Yurii Shyrma (iuriish@yahoo.com)
//
#include <helpers/HessenbergAndSchur.h>
#include <helpers/hhSequence.h>
#include <helpers/householder.h>
#include <helpers/jacobiSVD.h>

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename T>
Hessenberg<T>::Hessenberg(NDArray* matrix) {
  if (matrix->rankOf() != 2) THROW_EXCEPTION("ops::helpers::Hessenberg constructor: input matrix must be 2D !");

  if (matrix->sizeAt(0) == 1) {
    std::vector<LongType> qShape = {1, 1};
    _Q = new NDArray(matrix->ordering(),qShape, matrix->dataType(), matrix->getContext());
    *_Q = 1;
    _H = matrix->dup(matrix->ordering());
    return;
  }

  if (matrix->sizeAt(0) != matrix->sizeAt(1))
    THROW_EXCEPTION("ops::helpers::Hessenberg constructor: input array must be 2D square matrix !");

  _H = matrix->dup(matrix->ordering());
  _Q = matrix->ulike();

  evalData();
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void Hessenberg<T>::evalData() {
  const int rows = _H->sizeAt(0);

  std::vector<LongType> coeffsShape = {rows - 1};
  NDArray hhCoeffs(_H->ordering(), coeffsShape, _H->dataType(), _H->getContext());

  // calculate _H
  for (LongType i = 0; i < rows - 1; ++i) {
    T coeff, norm;

    NDArray hRef = *_H;
    NDArray tail1 = hRef({i + 1, -1, i, i + 1});
    NDArray tail2 = hRef({i + 2, -1, i, i + 1}, true);

    Householder<T>::evalHHmatrixDataI(tail1, coeff, norm);

    hRef({0, 0, i, i + 1}).template r<T>(i + 1) = norm;
    hhCoeffs.template r<T>(i) = coeff;

    NDArray bottomRightCorner = hRef({i + 1, -1, i + 1, -1}, true);
    Householder<T>::mulLeft(bottomRightCorner, tail2, coeff);
    NDArray tail2Trans = tail2.transpose();
    NDArray rightCols = hRef({0, 0, i + 1, -1}, true);
    Householder<T>::mulRight(rightCols, tail2Trans, coeff);
  }

  // calculate _Q
  HHsequence hhSeq(_H, &hhCoeffs, 'u');
  hhSeq._diagSize = rows - 1;
  hhSeq._shift = 1;
  hhSeq.applyTo_<T>(_Q);

  // fill down with zeros starting at first subdiagonal
  _H->fillAsTriangular<T>(0, -1, -1, *_H, 'l',false);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
Schur<T>::Schur(NDArray& matrix) {
  if (matrix.rankOf() != 2) THROW_EXCEPTION("ops::helpers::Schur constructor: input matrix must be 2D !");

  if (matrix.sizeAt(0) != matrix.sizeAt(1))
    THROW_EXCEPTION("ops::helpers::Schur constructor: input array must be 2D square matrix !");

  evalData(matrix);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void Schur<T>::evalData(NDArray& matrix) {
  const T scale = matrix.reduceNumber(reduce::AMax).template t<T>(0);


  if (scale < DataTypeUtils::min_positive<T>()) {
    t = matrix.ulike();
    u = matrix.ulike();

    t->nullify();
    u->setIdentity();

    return;
  }

  // perform Hessenberg decomposition
  NDArray matrixScale = matrix / scale;
  Hessenberg<T> hess(&matrixScale);

  t = hess._H;
  u = hess._Q;

  calcFromHessenberg();

  *t *= scale;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void Schur<T>::splitTwoRows(const int ind, const T shift) {
  const int numCols = t->sizeAt(1);

  T p = (T)0.5 * (t->t<T>(ind - 1, ind - 1) - t->t<T>(ind, ind));

  T q = p * p + t->t<T>(ind, ind - 1) * t->t<T>(ind - 1, ind);

  t->r<T>(ind, ind) += shift;
  t->r<T>(ind - 1, ind - 1) += shift;

  if (q >= (T)0) {
    T z = math::sd_sqrt<T, T>(math::sd_abs<T,T>(q));

    std::vector<LongType> rotShape = {2, 2};
    NDArray rotation(t->ordering(), rotShape, t->dataType(), t->getContext());

    if (p >= (T)0)
      JacobiSVD<T>::createJacobiRotationGivens(p + z, t->t<T>(ind, ind - 1), rotation);
    else
      JacobiSVD<T>::createJacobiRotationGivens(p - z, t->t<T>(ind, ind - 1), rotation);

    NDArray tRef = *t;
    NDArray rightCols = tRef({0, 0, ind - 1, -1});
    NDArray rotT = rotation.transpose();
    JacobiSVD<T>::mulRotationOnLeft(ind - 1, ind, rightCols, rotT);

    NDArray topRows = tRef({0, ind + 1, 0, 0});
    JacobiSVD<T>::mulRotationOnRight(ind - 1, ind, topRows, rotation);

    JacobiSVD<T>::mulRotationOnRight(ind - 1, ind, *u, rotation);

    t->r<T>(ind, ind - 1) = (T)0;
  }

  if (ind > 1) t->r<T>(ind - 1, ind - 2) = (T)0;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void Schur<T>::calcShift(const int ind, const int iter, T& shift, NDArray& shiftVec) {
  // shiftVec has length = 3

  shiftVec.r<T>(0) = t->t<T>(ind, ind);
  shiftVec.r<T>(1) = t->t<T>(ind - 1, ind - 1);
  shiftVec.r<T>(2) = t->t<T>(ind, ind - 1) * t->t<T>(ind - 1, ind);

  if (iter == 10) {
    shift += shiftVec.t<T>(0);

    for (int i = 0; i <= ind; ++i) t->r<T>(i, i) -= shiftVec.t<T>(0);

    T s = math::sd_abs<T,T>(t->t<T>(ind, ind - 1)) + math::sd_abs<T,T>(t->t<T>(ind - 1, ind - 2));

    shiftVec.r<T>(0) = T(0.75) * s;
    shiftVec.r<T>(1) = T(0.75) * s;
    shiftVec.r<T>(2) = T(-0.4375) * s * s;
  }

  if (iter == 30) {
    T s = (shiftVec.t<T>(1) - shiftVec.t<T>(0)) / T(2.0);
    s = s * s + shiftVec.t<T>(2);

    if (s > T(0)) {
      s = math::sd_sqrt<T, T>(s);

      if (shiftVec.t<T>(1) < shiftVec.t<T>(0)) s = -s;

      s = s + (shiftVec.t<T>(1) - shiftVec.t<T>(0)) / T(2.0);
      s = shiftVec.t<T>(0) - shiftVec.t<T>(2) / s;
      shift += s;

      for (int i = 0; i <= ind; ++i) t->r<T>(i, i) -= s;

      shiftVec = T(0.964);
    }
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void Schur<T>::initFrancisQR(const int ind1, const int ind2, NDArray& shiftVec, int& ind3,
                             NDArray& householderVec) {
  // shiftVec has length = 3

  for (ind3 = ind2 - 2; ind3 >= ind1; --ind3) {
    const T mm = t->t<T>(ind3, ind3);
    const T r = shiftVec.t<T>(0) - mm;
    const T s = shiftVec.t<T>(1) - mm;

    householderVec.r<T>(0) = (r * s - shiftVec.t<T>(2)) / t->t<T>(ind3 + 1, ind3) + t->t<T>(ind3, ind3 + 1);
    householderVec.r<T>(1) = t->t<T>(ind3 + 1, ind3 + 1) - mm - r - s;
    householderVec.r<T>(2) = t->t<T>(ind3 + 2, ind3 + 1);

    if (ind3 == ind1) break;

    const T lhs =
        t->t<T>(ind3, ind3 - 1) * (math::sd_abs<T,T>(householderVec.t<T>(1)) + math::sd_abs<T,T>(householderVec.t<T>(2)));
    const T rhs = householderVec.t<T>(0) * (math::sd_abs<T,T>(t->t<T>(ind3 - 1, ind3 - 1)) + math::sd_abs<T,T>(mm) +
                                            math::sd_abs<T,T>(t->t<T>(ind3 + 1, ind3 + 1)));

    if (math::sd_abs<T,T>(lhs) < DataTypeUtils::eps<T>() * rhs) break;
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void Schur<T>::doFrancisQR(const int ind1, const int ind2, const int ind3, NDArray& householderVec) {
  if (!(ind2 >= ind1))
    THROW_EXCEPTION(
        "ops::helpers::Schur:doFrancisQR: wrong input indexes, condition ind2 >= ind1 must be true !");
  if (!(ind2 <= ind3 - 2))
    THROW_EXCEPTION(
        "ops::helpers::Schur:doFrancisQR: wrong input indexes, condition iind2 <= ind3-2 must be true !");

  const int numCols = t->sizeAt(1);
  NDArray tRef = *t;
  NDArray uRef = *u;
  for (int k = ind2; k <= ind3 - 2; ++k) {
    const bool firstIter = (k == ind2);

    T coeff, normX;
    std::vector<LongType> tailShape = {2,1};
    NDArray tail(t->ordering(),tailShape, t->dataType(), t->getContext());
    NDArray first = firstIter ? householderVec : tRef({k, k + 3, k - 1, k});
    Householder<T>::evalHHmatrixData(first, tail, coeff, normX);

    if (normX != T(0)) {
      if (firstIter && k > ind1)
        t->r<T>(k, k - 1) = -t->t<T>(k, k - 1);
      else if (!firstIter)
        t->r<T>(k, k - 1) = normX;

      NDArray block1 = tRef({k, k + 3, k, numCols}, true);
      Householder<T>::mulLeft(block1, tail, coeff);

      NDArray block2 = tRef({0, math::sd_min<int>(ind3, k + 3) + 1, k, k + 3}, true);
      Householder<T>::mulRight(block2, tail, coeff);

      NDArray block3 = uRef({0, numCols, k, k + 3}, true);
      Householder<T>::mulRight(block3, tail, coeff);
    }
  }

  T coeff, normX;
  std::vector<LongType> tailShape = {1,1};
  NDArray tail(t->ordering(), tailShape, t->dataType(), t->getContext());
  NDArray first = tRef({ind3 - 1, ind3 + 1, ind3 - 2, ind3 - 1});
  Householder<T>::evalHHmatrixData(first, tail, coeff, normX);

  if (normX != T(0)) {
    t->r<T>(ind3 - 1, ind3 - 2) = normX;

    NDArray block1 = tRef({ind3 - 1, ind3 + 1, ind3 - 1, numCols}, true);
    Householder<T>::mulLeft(block1, tail, coeff);

    NDArray block2 = tRef({0, ind3 + 1, ind3 - 1, ind3 + 1}, true);
    Householder<T>::mulRight(block2, tail, coeff);

    NDArray block3 = uRef({0, numCols, ind3 - 1, ind3 + 1}, true);
    Householder<T>::mulRight(block3, tail, coeff);
  }

  for (int i = ind2 + 2; i <= ind3; ++i) {
    t->r<T>(i, i - 2) = T(0);
    if (i > ind2 + 2) t->r<T>(i, i - 3) = T(0);
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void Schur<T>::calcFromHessenberg() {
  const int maxIters = _maxItersPerRow * t->sizeAt(0);

  const int numCols = t->sizeAt(1);
  int iu = numCols - 1;
  int iter = 0;
  int totalIter = 0;

  T shift = T(0);
  NDArray tRef = *t;
  NDArray uRef = *u;
  T norm = static_cast<T>(0);
  for (int j = 0; j < numCols; ++j)
    norm += tRef({0, math::sd_min<int>(numCols, j + 2), j, j + 1}).reduceNumber(reduce::ASum).template t<T>(0);

  if (norm != T(0)) {
    while (iu >= 0) {
      const int il = getSmallSubdiagEntry(iu);

      if (il == iu) {
        t->r<T>(iu, iu) = t->t<T>(iu, iu) + shift;
        if (iu > 0) t->r<T>(iu, iu - 1) = T(0);
        iu--;
        iter = 0;

      } else if (il == iu - 1) {
        splitTwoRows(iu, shift);
        iu -= 2;
        iter = 0;
      } else {
        std::vector<LongType> shiftVecShape = {3};
        NDArray householderVec(t->ordering(), shiftVecShape, t->dataType(), t->getContext());
        NDArray shiftVec(t->ordering(), shiftVecShape, t->dataType(), t->getContext());

        calcShift(iu, iter, shift, shiftVec);

        ++iter;
        ++totalIter;

        if (totalIter > maxIters) break;

        int im;
        initFrancisQR(il, iu, shiftVec, im, householderVec);
        doFrancisQR(il, im, iu, householderVec);
      }
    }
  }
}

BUILD_SINGLE_TEMPLATE( class Hessenberg, , SD_FLOAT_TYPES);
BUILD_SINGLE_TEMPLATE( class Schur, , SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
