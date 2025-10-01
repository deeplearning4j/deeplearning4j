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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 03.01.2018
//
#include <array/ResultSet.h>
#include <helpers/biDiagonalUp.h>
#include <helpers/jacobiSVD.h>
#include <helpers/svd.h>

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename T>
SVD<T>::SVD(NDArray& matrix, const int switchSize, const bool calcU, const bool calcV, const bool fullUV) {
  if (matrix.rankOf() != 2 || matrix.isScalar())
    THROW_EXCEPTION("ops::helpers::SVD constructor: input array must be 2D matrix !");

  const int rows = matrix.sizeAt(0);
  const int cols = matrix.sizeAt(1);

  if (cols > rows) {
    _transp = true;
    _diagSize = rows;
  } else {
    _transp = false;
    _diagSize = cols;
  }

  _switchSize = switchSize;
  _calcU = calcU;
  _calcV = calcV;
  _fullUV = fullUV;

  if (_transp) math::sd_swap<bool>(_calcU, _calcV);
  std::vector<sd::LongType> sShape = {_diagSize, 1};
  std::vector<sd::LongType> mShape = {_diagSize + 1, _diagSize};
  _s = NDArray(matrix.ordering(), sShape, matrix.dataType(), matrix.getContext());
  _m = NDArray(matrix.ordering(), mShape, matrix.dataType(), matrix.getContext());
  std::vector<sd::LongType> uShapeOne = {_diagSize + 1, _diagSize + 1};
  std::vector<sd::LongType> uShapeTwo = {2, _diagSize + 1};
  if (_calcU)
    _u = NDArray(matrix.ordering(), uShapeOne, matrix.dataType(), matrix.getContext());
  else
    _u = NDArray(matrix.ordering(), uShapeTwo, matrix.dataType(), matrix.getContext());

  if (_calcV) {
    std::vector<sd::LongType> vShape = {_diagSize, _diagSize};
    _v = NDArray(matrix.ordering(),vShape, matrix.dataType(), matrix.getContext());
  }


  evalData(matrix);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
SVD<T>::SVD(NDArray& matrix, const int switchSize, const bool calcU, const bool calcV, const bool fullUV,
            const char t) {
  if (matrix.rankOf() != 2 || matrix.isScalar())
    THROW_EXCEPTION("ops::helpers::SVD constructor: input array must be 2D matrix !");

  const int rows = matrix.sizeAt(0);
  const int cols = matrix.sizeAt(1);

  if (cols > rows) {
    _transp = true;
    _diagSize = rows;
  } else {
    _transp = false;
    _diagSize = cols;
  }

  _switchSize = switchSize;
  _calcU = calcU;
  _calcV = calcV;
  _fullUV = fullUV;

  if (_transp) math::sd_swap<bool>(_calcU, _calcV);
  std::vector<sd::LongType> sShape = {_diagSize, 1};
  std::vector<sd::LongType> mShape = {_diagSize + 1, _diagSize};

  _s = NDArray(matrix.ordering(), sShape, matrix.dataType(), matrix.getContext());
  _m = NDArray(matrix.ordering(), mShape, matrix.dataType(), matrix.getContext());
  std::vector<sd::LongType> uShapeOne = {_diagSize + 1, _diagSize + 1};
  std::vector<sd::LongType> uShapeTwo = {2, _diagSize + 1};

  if (_calcU)
    _u = NDArray(matrix.ordering(), uShapeOne, matrix.dataType(), matrix.getContext());
  else
    _u = NDArray(matrix.ordering(), uShapeTwo, matrix.dataType(), matrix.getContext());

  if (_calcV) {
    std::vector<sd::LongType> vShape = {_diagSize, _diagSize};
    _v = NDArray(matrix.ordering(),vShape, matrix.dataType(), matrix.getContext());
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void SVD<T>::deflation1(int col1, int shift, int ind, int size) {
  if (ind <= 0)
    THROW_EXCEPTION("ops::helpers::SVD::deflation1 method: input int must satisfy condition ind > 0 !");

  int first = col1 + shift;
  T cos = _m.t<T>(first, first);
  T sin = _m.t<T>(first + ind, first);
  T denom = math::sd_sqrt<T, T>(cos * cos + sin * sin);

  if (denom == (T)0.) {
    _m.template r<T>(first + ind, first + ind) = (T)0;
    return;
  }

  cos /= denom;
  sin /= denom;

  _m.template r<T>(first, first) = denom;
  _m.template r<T>(first + ind, first) = (T)0;
  _m.template r<T>(first + ind, first + ind) = (T)0;

  std::vector<sd::LongType> rotShape = {2, 2};
  NDArray rotation(_m.ordering(), rotShape, _m.dataType(), _m.getContext());

  rotation.template r<T>(0, 0) = rotation.template r<T>(1, 1) = cos;
  rotation.template r<T>(0, 1) = -sin;
  rotation.template r<T>(1, 0) = sin;

  if (_calcU) {
    auto temp = _u({col1, col1 + size + 1, 0, 0}, true);
    JacobiSVD<T>::mulRotationOnRight(col1, col1 + ind, temp, rotation);
  } else
    JacobiSVD<T>::mulRotationOnRight(col1, col1 + ind, _u, rotation);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void SVD<T>::deflation2(int col1U, int col1M, int row1W, int col1W, int ind1, int ind2, int size) {
  if (ind1 >= ind2)
    THROW_EXCEPTION("ops::helpers::SVD::deflation2 method: input intes must satisfy condition ind1 < ind2 !");

  if (size <= 0)
    THROW_EXCEPTION("ops::helpers::SVD::deflation2 method: input size must satisfy condition size > 0 !");

  T cos = _m.t<T>(col1M + ind1, col1M);
  T sin = _m.t<T>(col1M + ind2, col1M);
  T denom = math::sd_sqrt<T, T>(cos * cos + sin * sin);

  if (denom == (T)0.) {
    _m.template r<T>(col1M + ind1, col1M + ind1) = _m.t<T>(col1M + ind2, col1M + ind2);
    return;
  }

  cos /= denom;
  sin /= denom;
  _m.template r<T>(col1M + ind1, col1M) = denom;
  _m.template r<T>(col1M + ind2, col1M + ind2) = _m.t<T>(col1M + ind1, col1M + ind1);
  _m.template r<T>(col1M + ind2, col1M) = (T)0;

  std::vector<sd::LongType> rotShape = {2, 2};
  NDArray rotation(_m.ordering(), rotShape, _m.dataType(), _m.getContext());

  rotation.template r<T>(0, 0) = rotation.template r<T>(1, 1) = cos;
  rotation.template r<T>(0, 1) = -sin;
  rotation.template r<T>(1, 0) = sin;

  if (_calcU) {
    auto temp = _u({col1U, col1U + size + 1, 0, 0}, true);
    JacobiSVD<T>::mulRotationOnRight(col1U + ind1, col1U + ind2, temp, rotation);
  } else
    JacobiSVD<T>::mulRotationOnRight(col1U + ind1, col1U + ind2, _u, rotation);

  if (_calcV) {
    auto temp = _v({row1W, row1W + size, 0, 0}, true);
    JacobiSVD<T>::mulRotationOnRight(col1W + ind1, col1W + ind2, temp, rotation);
  }
}

//////////////////////////////////////////////////////////////////////////
// has effect on block from (col1+shift, col1+shift) to (col2+shift, col2+shift) inclusively
template <typename T>
void SVD<T>::deflation(int col1, int col2, int ind, int row1W, int col1W, int shift) {
  const int len = col2 + 1 - col1;

  NDArray colVec0 = _m({col1 + shift, col1 + shift + len, col1 + shift, col1 + shift + 1}, true);

  NDArray diagInterval = _m({col1 + shift, col1 + shift + len, col1 + shift, col1 + shift + len}, true).diagonal('c');

  const T almostZero = DataTypeUtils::min_positive<T>();
  T maxElem;
  if (len == 1)
    maxElem = math::sd_abs<T,T>(diagInterval.template t<T>(0));
  else
    maxElem = diagInterval({1, -1, 0, 0}, true).reduceNumber(reduce::AMax).template t<T>(0);
  T maxElem0 = colVec0.reduceNumber(reduce::AMax).template t<T>(0);

  T eps = math::sd_max<T>(almostZero, DataTypeUtils::eps<T>() * maxElem);
  T epsBig = (T)8. * DataTypeUtils::eps<T>() * math::sd_max<T>(maxElem0, maxElem);

  if (diagInterval.template t<T>(0) < epsBig) diagInterval.template r<T>(0) = epsBig;

  for (int i = 1; i < len; ++i)
    if (math::sd_abs<T,T>(colVec0.template t<T>(i)) < eps) colVec0.template r<T>(i) = (T)0;

  for (int i = 1; i < len; i++)
    if (diagInterval.template t<T>(i) < epsBig) {
      deflation1(col1, shift, i, len);
      for (int j = 0; j < len; ++j) diagInterval.template r<T>(j) = _m.t<T>(col1 + shift + j, col1 + shift + j);
    }

  {
    bool totDefl = true;
    for (int i = 1; i < len; i++)
      if (colVec0.template t<T>(i) >= almostZero) {
        totDefl = false;
        break;
      }

    int* permut = nullptr;
    ALLOCATE(permut, _m.getContext()->getWorkspace(), 3 * _diagSize, int);
    {
      permut[0] = 0;
      int p = 1;

      for (int i = 1; i < len; ++i)
        if (math::sd_abs<T,T>(diagInterval.template t<T>(i)) < almostZero) permut[p++] = i;

      int k = 1, m = ind + 1;

      for (; p < len; ++p) {
        if (k > ind)
          permut[p] = m++;
        else if (m >= len)
          permut[p] = k++;
        else if (diagInterval.template t<T>(k) < diagInterval.template t<T>(m))
          permut[p] = m++;
        else
          permut[p] = k++;
      }
    }

    if (totDefl) {
      for (int i = 1; i < len; ++i) {
        int ki = permut[i];
        if (math::sd_abs<T,T>(diagInterval.template t<T>(ki)) < almostZero ||
            diagInterval.template t<T>(0) < diagInterval.template t<T>(ki))
          permut[i - 1] = permut[i];
        else {
          permut[i - 1] = 0;
          break;
        }
      }
    }

    int* tInd = permut + len;
    int* tCol = permut + 2 * len;

    for (int m = 0; m < len; m++) {
      tCol[m] = m;
      tInd[m] = m;
    }

    for (int i = totDefl ? 0 : 1; i < len; i++) {
      const int ki = permut[len - (totDefl ? i + 1 : i)];
      const int jac = tCol[ki];

      math::sd_swap<T>(diagInterval.template r<T>(i), diagInterval.template r<T>(jac));

      if (i != 0 && jac != 0) math::sd_swap<T>(colVec0.template r<T>(i), colVec0.template r<T>(jac));

      if (_calcU) {
        auto temp1 = _u({col1, col1 + len + 1, col1 + i, col1 + i + 1});
        auto temp2 = _u({col1, col1 + len + 1, col1 + jac, col1 + jac + 1});
        temp1.swapUnsafe(temp2);
      } else {
        auto temp1 = _u({0, 2, col1 + i, col1 + i + 1});
        auto temp2 = _u({0, 2, col1 + jac, col1 + jac + 1});
        temp1.swapUnsafe(temp2);
      }

      if (_calcV) {
        auto temp1 = _v({row1W, row1W + len, col1W + i, col1W + i + 1});
        auto temp2 = _v({row1W, row1W + len, col1W + jac, col1W + jac + 1});
        temp1.swapUnsafe(temp2);
      }

      const int tI = tInd[i];
      tCol[tI] = jac;
      tCol[ki] = i;
      tInd[jac] = tI;
      tInd[i] = ki;
    }

    RELEASE(permut, _m.getContext()->getWorkspace());
  }

  {
    int i = len - 1;

    while (i > 0 && (math::sd_abs<T,T>(diagInterval.template t<T>(i)) < almostZero ||
                     math::sd_abs<T,T>(colVec0.template t<T>(i)) < almostZero))
      --i;

    for (; i > 1; --i) {
      if ((diagInterval.template t<T>(i) - diagInterval.template t<T>(i - 1)) < DataTypeUtils::eps<T>() * maxElem) {
        if (math::sd_abs<T,T>(diagInterval.template t<T>(i) - diagInterval.template t<T>(i - 1)) >= epsBig)
          THROW_EXCEPTION("ops::helpers::SVD::deflation: diagonal elements are not properly sorted !");
        deflation2(col1, col1 + shift, row1W, col1W, i - 1, i, len);
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
T SVD<T>::secularEq(const T diff, NDArray& col0, NDArray& diag, NDArray permut,
                    NDArray& diagShifted, const T shift) {
  auto len = permut.lengthOf();
  T res = static_cast<T>(1.);
  T item;
  for (int i = 0; i < len; ++i) {
    int j = (int)permut.t<T>(i);
    item = col0.t<T>(j) / ((diagShifted.t<T>(j) - diff) * (diag.t<T>(j) + shift + diff));
    res += item * col0.t<T>(j);
  }

  return res;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void SVD<T>::calcSingVals(NDArray col0, NDArray& diag, NDArray& permut, NDArray& singVals,
                          NDArray& shifts, NDArray& mus) {
  auto len = col0.lengthOf();
  auto curLen = len;

  while (curLen > 1 && col0.t<T>(curLen - 1) == (T)0.f) --curLen;

  for (sd::LongType k = 0; k < len; ++k) {
    if (col0.t<T>(k) == (T)0.f || curLen == 1) {
      singVals.template r<T>(k) = k == 0 ? col0.t<T>(0) : diag.t<T>(k);
      mus.template r<T>(k) = (T)0;
      shifts.template r<T>(k) = k == 0 ? col0.t<T>(0) : diag.t<T>(k);
      continue;
    }

    T left = diag.t<T>(k);
    T right;

    if (k == curLen - 1)
      right = diag.t<T>(curLen - 1) + col0.reduceNumber(reduce::Norm2).t<T>(0);
    else {
      int l = k + 1;
      while (col0.t<T>(l) == (T)0.f) {
        ++l;
        if (l >= curLen) THROW_EXCEPTION("ops::helpers::SVD::calcSingVals method: l >= curLen !");
      }

      right = diag.t<T>(l);
    }

    T mid = left + (right - left) / (T)2.;
    T fMid = secularEq(mid, col0, diag, permut, diag, static_cast<T>(0.));
    T shift = (k == curLen - 1 || fMid > (T)0.) ? left : right;

    auto diagShifted = diag - shift;

    T muPrev, muCur;
    if (shift == left) {
      muPrev = (right - left) * 0.1;
      if (k == curLen - 1)
        muCur = right - left;
      else
        muCur = (right - left) * 0.5;
    } else {
      muPrev = -(right - left) * 0.1;
      muCur = -(right - left) * 0.5;
    }

    T fPrev = secularEq(muPrev, col0, diag, permut, diagShifted, shift);
    T fCur = secularEq(muCur, col0, diag, permut, diagShifted, shift);

    if (math::sd_abs<T,T>(fPrev) < math::sd_abs<T,T>(fCur)) {
      math::sd_swap<T>(fPrev, fCur);
      math::sd_swap<T>(muPrev, muCur);
    }

    bool useBisection = fPrev * fCur > (T)0.;
    while (fCur != (T).0 &&
           math::sd_abs<T,T>(muCur - muPrev) >
           (T)8. * DataTypeUtils::eps<T>() * math::sd_max<T>(math::sd_abs<T,T>(muCur), math::sd_abs<T,T>(muPrev)) &&
           math::sd_abs<T,T>(fCur - fPrev) > DataTypeUtils::eps<T>() && !useBisection) {
      T a = (fCur - fPrev) / ((T)1. / muCur - (T)1. / muPrev);
      T jac = fCur - a / muCur;
      T muZero = -a / jac;
      T fZero = secularEq(muZero, col0, diag, permut, diagShifted, shift);

      muPrev = muCur;
      fPrev = fCur;
      muCur = muZero;
      fCur = fZero;

      if (shift == left && (muCur < (T)0. || muCur > right - left))
        useBisection = true;
      else if (shift == right && (muCur < -(right - left) || muCur > (T)0.))
        useBisection = true;
      else if (math::sd_abs<T,T>(fCur) > math::sd_abs<T,T>(fPrev) &&
               math::sd_abs<T,T>(fCur - fPrev) > (T)16. * DataTypeUtils::eps<T>())
        useBisection = true;
    }

    if (useBisection) {
      T leftShifted, rightShifted;
      if (shift == left) {
        leftShifted = DataTypeUtils::min_positive<T>();
        rightShifted = (k == curLen - 1) ? right : ((right - left) * (T)0.6);
      } else {
        leftShifted = -(right - left) * (T)0.6;
        rightShifted = -DataTypeUtils::min_positive<T>();
      }

      T fLeft = secularEq(leftShifted, col0, diag, permut, diagShifted, shift);

      while (rightShifted - leftShifted >
             (T)2.f * DataTypeUtils::eps<T>() *
             math::sd_max<T>(math::sd_abs<T,T>(leftShifted), math::sd_abs<T,T>(rightShifted))) {
        T midShifted = (leftShifted + rightShifted) / (T)2.;
        fMid = secularEq(midShifted, col0, diag, permut, diagShifted, shift);
        if (fLeft * fMid < (T)0.)
          rightShifted = midShifted;
        else {
          leftShifted = midShifted;
          fLeft = fMid;
        }
      }
      muCur = (leftShifted + rightShifted) / (T)2.;
    }
    singVals.template r<T>(k) = shift + muCur;
    shifts.template r<T>(k) = shift;
    mus.template r<T>(k) = muCur;
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void SVD<T>::perturb(NDArray col0, NDArray& diag, NDArray permut, NDArray& singVals,
                     NDArray& shifts, NDArray& mus, NDArray& zhat) {
  int n = col0.lengthOf();
  int m = permut.lengthOf();
  if (m == 0) {
    zhat.nullify();
    return;
  }

  int last = permut.t<T>(m - 1);

  for (int k = 0; k < n; ++k) {
    if (col0.t<T>(k) == (T)0.f)
      zhat.template r<T>(k) = (T)0;
    else {
      T dk = diag.t<T>(k);
      T prod = (singVals.t<T>(last) + dk) * (mus.t<T>(last) + (shifts.t<T>(last) - dk));

      for (int l = 0; l < m; ++l) {
        int i = (int)permut.t<T>(l);
        if (i != k) {
          int j = i < k ? i : (int)permut.t<T>(l - 1);
          prod *= ((singVals.t<T>(j) + dk) / ((diag.t<T>(i) + dk))) *
                  ((mus.t<T>(j) + (shifts.t<T>(j) - dk)) / ((diag.t<T>(i) - dk)));
        }
      }
      T tmp = math::sd_sqrt<T, T>(prod);
      zhat.template r<T>(k) = col0.t<T>(k) > (T)0 ? tmp : -tmp;
    }
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void SVD<T>::calcSingVecs(NDArray zhat, NDArray& diag, NDArray perm, NDArray& singVals,
                          NDArray& shifts, NDArray& mus, NDArray& U, NDArray& V) {
  int n = zhat.lengthOf();
  int m = perm.lengthOf();

  for (int k = 0; k < n; ++k) {
    NDArray colU = U({0, 0, k, k + 1});
    colU.nullify();

    NDArray colV;

    if (_calcV) {
      colV = V({0, 0, k, k + 1});
      colV.nullify();
    }

    if (zhat.t<T>(k) == (T)0.f) {
      colU.template r<T>(k) = (T)1;

      if (_calcV) colV.template r<T>(k) = (T)1;
    } else {
      for (int l = 0; l < m; ++l) {
        int i = (int)perm.t<T>(l);
        U.template r<T>(i, k) =
            zhat.t<T>(i) / (((diag.t<T>(i) - shifts.t<T>(k)) - mus.t<T>(k))) / ((diag.t<T>(i) + singVals.t<T>(k)));
      }
      U.template r<T>(n, k) = (T)0;
      colU /= colU.reduceNumber(reduce::Norm2);

      if (_calcV) {
        for (int l = 1; l < m; ++l) {
          int i = perm.t<T>(l);
          V.template r<T>(i, k) = diag.t<T>(i) * zhat.t<T>(i) / (((diag.t<T>(i) - shifts.t<T>(k)) - mus.t<T>(k))) /
                                  ((diag.t<T>(i) + singVals.t<T>(k)));
        }
        V.template r<T>(0, k) = (T)-1;
        colV /= colV.reduceNumber(reduce::Norm2);
      }
    }
  }

  NDArray colU = U({0, 0, n, n + 1});
  colU.nullify();
  colU.template r<T>(n) = (T)1;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void SVD<T>::calcBlockSVD(int col1, int size, NDArray& U, NDArray& singVals, NDArray& V) {

  const T almostZero = DataTypeUtils::min_positive<T>();
  int end = col1 + size;
  auto col0 = _m({col1, end, col1, col1 + 1}, true);
  auto view = _m({col1, end, col1, end}, true);
  auto diag = view.diagonal('c');
  diag.template r<T>(0) = (T)0;
  std::vector<sd::LongType> shape2 = {size, 1};
  std::vector<sd::LongType> shape3 = {size + 1, size + 1};
  singVals = NDArray(_m.ordering(), shape2, _m.dataType(), _m.getContext());
  U = NDArray(_u.ordering(), shape3, _u.dataType(), _u.getContext());
  std::vector<sd::LongType> sizeShape = {size, size};
  if (_calcV) V = NDArray(_v.ordering(), sizeShape, _v.dataType(), _v.getContext());

  int curSize = size;
  while (curSize > 1 && diag.template t<T>(curSize - 1) == (T)0.f) --curSize;

  int m = 0;
  std::vector<int> indices;
  for (int k = 0; k < curSize; ++k)
    if (math::sd_abs<T,T>(col0.template t<T>(k)) > almostZero) indices.push_back(k);

  std::vector<sd::LongType> permutShape = {(int)indices.size()};
  NDArray permut(_m.ordering(), permutShape, _m.dataType(), _m.getContext());
  for (size_t k = 0; k < indices.size(); ++k) permut.template r<T>(k) = (T)indices[k];

  std::vector<sd::LongType> shape = {size,1};
  NDArray shifts(_m.ordering(), shape, _m.dataType(), _m.getContext());
  NDArray mus(_m.ordering(), shape, _m.dataType(), _m.getContext());
  NDArray zhat(_m.ordering(),shape, _m.dataType(), _m.getContext());

  calcSingVals(col0, diag, permut, singVals, shifts, mus);
  perturb(col0, diag, permut, singVals, shifts, mus, zhat);
  calcSingVecs(zhat, diag, permut, singVals, shifts, mus, U, V);

  for (int i = 0; i < curSize - 1; ++i) {
    if (singVals.t<T>(i) > singVals.t<T>(i + 1)) {
      math::sd_swap<T>(singVals.template r<T>(i), singVals.template r<T>(i + 1));

      auto temp1 = U({0, 0, i, i + 1});
      auto temp2 = U({0, 0, i + 1, i + 2});
      temp1.swapUnsafe(temp2);

      if (_calcV) {
        auto temp1V = V({0, 0, i, i + 1});
        auto temp2V = V({0, 0, i + 1, i + 2});
        temp1V.swapUnsafe(temp2V);
      }
    }
  }

  auto temp1 = singVals({0, curSize, 0, 0});
  for (int e = 0; e < curSize / 2; ++e) math::sd_swap<T>(temp1.template r<T>(e), temp1.template r<T>(curSize - 1 - e));

  auto temp2 = U({0, 0, 0, curSize}, true);
  for (int i = 0; i < curSize / 2; ++i) {
    auto temp3 = temp2({0, 0, i, i + 1});
    auto temp4 = temp2({0, 0, curSize - 1 - i, curSize - i});
    temp3.swapUnsafe(temp4);
  }


  if (_calcV) {
    auto temp2V = V({0, 0, 0, curSize}, true);
    for (int i = 0; i < curSize / 2; ++i) {
      auto temp3 = temp2V({0, 0, i, i + 1});
      auto temp4 = temp2V({0, 0, curSize - 1 - i, curSize - i});
      temp3.swapUnsafe(temp4);
    }
  }

}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void SVD<T>::DivideAndConquer(int col1, int col2, int row1W, int col1W, int shift) {
  // requires rows = cols + 1;
  const int n = col2 - col1 + 1;
  const int k = n / 2;
  const T almostZero = DataTypeUtils::min_positive<T>();
  T alphaK, betaK, r0, lambda, phi, c0, s0;

  std::vector<sd::LongType> lShape = {1, k};
  std::vector<sd::LongType> fShape = {1, n - k - 1};
  NDArray l(_u.ordering(),lShape, _u.dataType(), _u.getContext());
  NDArray f(_u.ordering(), fShape, _u.dataType(), _u.getContext());

  if (n < _switchSize) {
    JacobiSVD<T> jac(_m({col1, col1 + n + 1, col1, col1 + n}, true), _calcU, _calcV, _fullUV);

    if (_calcU)
      _u({col1, col1 + n + 1, col1, col1 + n + 1}, true).assign(&jac._u);
    else {
      _u({0, 1, col1, col1 + n + 1}, true).assign(&jac._u({0, 1, 0, 0}, true));
      _u({1, 2, col1, col1 + n + 1}, true).assign(&jac._u({n, n + 1, 0, 0}, true));
    }

    if (_calcV) _v({row1W, row1W + n, col1W, col1W + n}, true).assign(&jac._v);

    _m({col1 + shift, col1 + shift + n + 1, col1 + shift, col1 + shift + n}, true).nullify();
    auto diag = _m.diagonal('c');
    NDArray first =  diag({col1 + shift, col1 + shift + n, 0, 0}, true);
    NDArray second = jac._s({0, n, 0, 0}, true);
    first.assign(&second);

    return;
  }

  alphaK = _m.t<T>(col1 + k, col1 + k);
  betaK = _m.t<T>(col1 + k + 1, col1 + k);

  DivideAndConquer(k + 1 + col1, col2, k + 1 + row1W, k + 1 + col1W, shift);
  DivideAndConquer(col1, k - 1 + col1, row1W, col1W + 1, shift + 1);

  if (_calcU) {
    lambda = _u.t<T>(col1 + k, col1 + k);
    phi = _u.t<T>(col1 + k + 1, col2 + 1);
  } else {
    lambda = _u.t<T>(1, col1 + k);
    phi = _u.t<T>(0, col2 + 1);
  }


  r0 = math::sd_sqrt<T, T>((math::sd_abs<T,T>(alphaK * lambda) * math::sd_abs<T,T>(alphaK * lambda)) +
                           math::sd_abs<T,T>(betaK * phi) * math::sd_abs<T,T>(betaK * phi));

  if (_calcU) {
    l.assign(&_u({col1 + k, col1 + k + 1, col1, col1 + k}, true));
    f.assign(&_u({col1 + k + 1, col1 + k + 2, col1 + k + 1, col1 + n}, true));
  } else {
    l.assign(&_u({1, 2, col1, col1 + k}, true));
    f.assign(&_u({0, 1, col1 + k + 1, col1 + n}, true));
  }


  if (_calcV) _v.template r<T>(row1W + k, col1W) = (T)1;

  if (r0 < almostZero) {
    c0 = 1.;
    s0 = 0.;
  } else {
    c0 = alphaK * lambda / r0;
    s0 = betaK * phi / r0;
  }


  if (_calcU) {
    NDArray *q1 = _u({col1, col1 + k + 1, col1 + k, col1 + k + 1}, true).dup();
    NDArray *q1Ref = q1;
    NDArray uAssignOne = *q1 * c0;
    NDArray uAssignTwo = *q1 * (-s0);
    for (int i = col1 + k - 1; i >= col1; --i)
      _u({col1, col1 + k + 1, i + 1, i + 2}, true).assign(&_u({col1, col1 + k + 1, i, i + 1}, true));

    NDArray temp1 = _u({col1 + k + 1, col1 + n + 1, col2 + 1, col2 + 2}, true);
    NDArray uAssignThree = temp1 * s0;


    _u({col1, col1 + k + 1, col1, col1 + 1}, true).assign(&uAssignOne);
    _u({col1, col1 + k + 1, col2 + 1, col2 + 2}, true).assign(&uAssignTwo);
    _u({col1 + k + 1, col1 + n + 1, col1, col1 + 1}, true).assign(&uAssignThree);

    temp1 *= c0;
  } else {
    T q1 = _u.t<T>(0, col1 + k);

    for (int i = col1 + k - 1; i >= col1; --i) _u.template r<T>(0, i + 1) = _u.template r<T>(0, i);

    _u.template r<T>(0, col1) = q1 * c0;
    _u.template r<T>(0, col2 + 1) = -q1 * s0;
    _u.template r<T>(1, col1) = _u.t<T>(1, col2 + 1) * s0;
    _u.template r<T>(1, col2 + 1) = _u.t<T>(1, col2 + 1) * c0;
    _u({1, 2, col1 + 1, col1 + k + 1}).nullify();
    _u({0, 1, col1 + k + 1, col1 + n}).nullify();
  }


  _m.template r<T>(col1 + shift, col1 + shift) = r0;

  NDArray assignOne = l * alphaK;
  NDArray assignTwo = f * betaK;
  _m({col1 + shift + 1, col1 + shift + k + 1, col1 + shift, col1 + shift + 1}, true).assign(&assignOne);
  _m({col1 + shift + k + 1, col1 + shift + n, col1 + shift, col1 + shift + 1}, true).assign(&assignTwo);

  deflation(col1, col2, k, row1W, col1W, shift);

  NDArray UofSVD, VofSVD, singVals;

  calcBlockSVD(col1 + shift, n, UofSVD, singVals, VofSVD);
  if (_calcU) {
    auto temp = _u({col1, col1 + n + 1, col1, col1 + n + 1}, true);
    NDArray assign2 = mmul(temp, UofSVD);
    temp.assign(&assign2);
  } else {
    auto temp = _u({0, 0, col1, col1 + n + 1}, true);
    NDArray assign2 = mmul(temp, UofSVD);
    temp.assign(&assign2);
  }

  if (_calcV) {
    auto temp = _v({row1W, row1W + n, row1W, row1W + n}, true);
    NDArray assign2 = mmul(temp, VofSVD);
    temp.assign(&assign2);
  }


  auto blockM = _m({col1 + shift, col1 + shift + n, col1 + shift, col1 + shift + n}, true);
  blockM.nullify();

  blockM.diagonal('c').assign(&singVals);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void SVD<T>::exchangeUV(HHsequence& hhU, HHsequence& hhV, NDArray& U, NDArray& V) {
  if (_calcU) {
    int colsU = _fullUV ? hhU.rows() : _diagSize;
    std::vector<sd::LongType> tempShape = {hhU.rows(), colsU};
    NDArray temp1(_u.ordering(), tempShape, _u.dataType(), _u.getContext());
    temp1.setIdentity();
    _u = temp1;

    _u({0, _diagSize, 0, _diagSize}, true).assign(&V({0, _diagSize, 0, _diagSize}, true));
    const_cast<HHsequence&>(hhU).mulLeft(&_u);
  }

  if (_calcV) {
    int colsV = _fullUV ? hhV.rows() : _diagSize;
    std::vector<sd::LongType> tempShape = {hhV.rows(), colsV};
    NDArray temp1(_v.ordering(), tempShape, _v.dataType(), _v.getContext());
    temp1.setIdentity();
    _v = temp1;

    NDArray assign = U({0, _diagSize, 0, _diagSize}, true);
    _v({0, _diagSize, 0, _diagSize}, true).assign(&assign);
    const_cast<HHsequence&>(hhV).mulLeft(&_v);
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void SVD<T>::evalData(NDArray& matrix) {
  const T almostZero = DataTypeUtils::min_positive<T>();

  if (matrix.sizeAt(1) < _switchSize) {
    JacobiSVD<T> jac(matrix, _calcU, _calcV, _fullUV);

    if (_calcU) _u = jac._u;
    if (_calcV) _v = jac._v;
    _s.assign(&jac._s);

    return;
  }

  T scale = matrix.reduceNumber(reduce::AMax).t<T>(0);

  if (scale == (T)0.) scale = 1.;
  NDArray input = _transp ? matrix.transpose() : matrix / scale;
  BiDiagonalUp biDiag(input);

  _u.nullify();
  _v.nullify();

  NDArray assign1 = biDiag._HHbidiag.transpose();
  _m({0, _diagSize, 0, 0}, true).assign(&assign1);

  _m({_m.sizeAt(0) - 1, _m.sizeAt(0), 0, 0}).nullify();

  DivideAndConquer(0, _diagSize - 1, 0, 0, 0);

  for (int i = 0; i < _diagSize; ++i) {
    T a = math::sd_abs<T,T>(_m.t<T>(i, i));
    _s.template r<T>(i) = a * scale;
    if (a < almostZero) {
      _s({i + 1, _diagSize, 0, 0}).nullify();
      break;
    } else if (i == _diagSize - 1)
      break;
  }

  HHsequence hhV = biDiag.makeHHsequence('v');
  HHsequence hhU = biDiag.makeHHsequence('u');

  if (_transp)
    exchangeUV(hhV, hhU, _v, _u);
  else
    exchangeUV(hhU, hhV, _u, _v);

}

BUILD_SINGLE_TEMPLATE( class SVD, , SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
