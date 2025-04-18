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
#include <helpers/EigenValsAndVecs.h>
#include <helpers/HessenbergAndSchur.h>

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename T>
EigenValsAndVecs<T>::EigenValsAndVecs(NDArray& matrix) {
  if (matrix.rankOf() != 2)
    THROW_EXCEPTION("ops::helpers::EigenValsAndVecs constructor: input matrix must be 2D !");

  if (matrix.sizeAt(0) != matrix.sizeAt(1))
    THROW_EXCEPTION("ops::helpers::EigenValsAndVecs constructor: input array must be 2D square matrix !");

  Schur<T> schur(matrix);

  NDArray& schurMatrixU = schur.u;
  NDArray& schurMatrixT = schur.t;

  std::vector<LongType> shape = {schurMatrixU.sizeAt(1), schurMatrixU.sizeAt(1), 2};
  _Vecs = NDArray(matrix.ordering(), shape, matrix.dataType(),
                  matrix.getContext());
  std::vector<LongType> shape2 = {matrix.sizeAt(1), 2};
  _Vals = NDArray(matrix.ordering(), shape2, matrix.dataType(), matrix.getContext());

  // sequence of methods calls matters
  calcEigenVals(schurMatrixT);
  calcPseudoEigenVecs(schurMatrixT, schurMatrixU);  // pseudo-eigenvectors are real and will be stored in schurMatrixU
  calcEigenVecs(schurMatrixU);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void calcEigenVals_(NDArray& schurMatrixT, NDArray& _Vals) {
  const int numOfCols = schurMatrixT.sizeAt(1);

  // calculate eigenvalues _Vals
  int i = 0;
  while (i < numOfCols) {
    if (i == numOfCols - 1 || schurMatrixT.t<T>(i + 1, i) == T(0.f)) {
      _Vals.r<T>(i, 0) = schurMatrixT.t<T>(i, i);  // real part
      _Vals.r<T>(i, 1) = T(0);                     // imaginary part

      if (!math::sd_isfin<T>(_Vals.t<T>(i, 0))) {
        THROW_EXCEPTION("ops::helpers::igenValsAndVec::calcEigenVals: got infinite eigen value !");
        return;
      }

      ++i;
    } else {
      T p = T(0.5) * (schurMatrixT.t<T>(i, i) - schurMatrixT.t<T>(i + 1, i + 1));
      T z;
      {
        T t0 = schurMatrixT.t<T>(i + 1, i);
        T t1 = schurMatrixT.t<T>(i, i + 1);
        T maxval = math::sd_max<T>(math::sd_abs<T,T>(p), math::sd_max<T>(math::sd_abs<T,T>(t0), math::sd_abs<T,T>(t1)));
        t0 /= maxval;
        t1 /= maxval;
        T p0 = p / maxval;
        z = maxval * math::sd_sqrt<T, T>(math::sd_abs<T,T>(p0 * p0 + t0 * t1));
      }

      _Vals.r<T>(i, 0) = _Vals.r<T>(i + 1, 0) = schurMatrixT.t<T>(i + 1, i + 1) + p;
      _Vals.r<T>(i, 1) = z;
      _Vals.r<T>(i + 1, 1) = -z;

      if (!(math::sd_isfin<T>(_Vals.t<T>(i, 0)) && math::sd_isfin<T>(_Vals.t<T>(i + 1, 0)) &&
            math::sd_isfin<T>(_Vals.t<T>(i, 1))) &&
          math::sd_isfin<T>(_Vals.t<T>(i + 1, 1))) {
        THROW_EXCEPTION("ops::helpers::igenValsAndVec::calcEigenVals: got infinite eigen value !");
        return;
      }

      i += 2;
    }
  }
}

template <typename T>
void EigenValsAndVecs<T>::calcEigenVals(NDArray& schurMatrixT) {
  calcEigenVals_<T>(schurMatrixT, _Vals);
}
//////////////////////////////////////////////////////////////////////////
template <typename T>
void calcPseudoEigenVecs_(NDArray& schurMatrixT, NDArray& schurMatrixU, NDArray& _Vals) {
  const int numOfCols = schurMatrixU.sizeAt(1);

  T norm = 0;
  for (int j = 0; j < numOfCols; ++j)
    norm += schurMatrixT({j, j + 1, math::sd_max<LongType>(j - 1, 0), numOfCols})
        .reduceNumber(reduce::ASum)
        .template t<T>(0);

  if (norm == T(0)) return;

  for (int n = numOfCols - 1; n >= 0; n--) {
    T p = _Vals.t<T>(n, 0);  // real part
    T q = _Vals.t<T>(n, 1);  // imaginary part

    if (q == (T)0) {  // not complex

      T lastr((T)0), lastw((T)0);
      int l = n;

      schurMatrixT.r<T>(n, n) = T(1);

      for (int i = n - 1; i >= 0; i--) {
        T w = schurMatrixT.t<T>(i, i) - p;
        T r = mmul(schurMatrixT({i, i + 1, l, n + 1}, true), schurMatrixT({l, n + 1, n, n + 1}, true))
            .template t<T>(0);  // dot

        if (_Vals.t<T>(i, 1) < T(0)) {
          lastw = w;
          lastr = r;
        } else {
          l = i;
          if (_Vals.t<T>(i, 1) == T(0)) {
            if (w != T(0))
              schurMatrixT.r<T>(i, n) = -r / w;
            else
              schurMatrixT.r<T>(i, n) = -r / (DataTypeUtils::eps<T>() * norm);
          } else {
            T x = schurMatrixT.t<T>(i, i + 1);
            T y = schurMatrixT.t<T>(i + 1, i);
            T denom = (_Vals.t<T>(i, 0) - p) * (_Vals.t<T>(i, 0) - p) + _Vals.t<T>(i, 1) * _Vals.t<T>(i, 1);
            T t = (x * lastr - lastw * r) / denom;
            schurMatrixT.r<T>(i, n) = t;

            if (math::sd_abs<T,T>(x) > math::sd_abs<T,T>(lastw))
              schurMatrixT.r<T>(i + 1, n) = (-r - w * t) / x;
            else
              schurMatrixT.r<T>(i + 1, n) = (-lastr - y * t) / lastw;
          }

          T t = math::sd_abs<T,T>(schurMatrixT.t<T>(i, n));
          if ((DataTypeUtils::eps<T>() * t) * t > T(1))
            schurMatrixT({schurMatrixT.sizeAt(0) - numOfCols + i, -1, n, n + 1}) /= t;
        }
      }
    } else if (q < T(0) && n > 0) {  // complex

      T lastra(0), lastsa(0), lastw(0);
      int l = n - 1;

      if (math::sd_abs<T,T>(schurMatrixT.t<T>(n, n - 1)) > math::sd_abs<T,T>(schurMatrixT.t<T>(n - 1, n))) {
        schurMatrixT.r<T>(n - 1, n - 1) = q / schurMatrixT.t<T>(n, n - 1);
        schurMatrixT.r<T>(n - 1, n) = -(schurMatrixT.t<T>(n, n) - p) / schurMatrixT.t<T>(n, n - 1);
      } else {
        EigenValsAndVecs<T>::divideComplexNums(T(0), -schurMatrixT.t<T>(n - 1, n), schurMatrixT.t<T>(n - 1, n - 1) - p,
                                               q, schurMatrixT.r<T>(n - 1, n - 1), schurMatrixT.r<T>(n - 1, n));
      }

      schurMatrixT.r<T>(n, n - 1) = T(0);
      schurMatrixT.r<T>(n, n) = T(1);

      for (int i = n - 2; i >= 0; i--) {
        T ra = mmul(schurMatrixT({i, i + 1, l, n + 1}, true), schurMatrixT({l, n + 1, n - 1, n}, true))
            .template t<T>(0);  // dot
        T sa = mmul(schurMatrixT({i, i + 1, l, n + 1}, true), schurMatrixT({l, n + 1, n, n + 1}, true))
            .template t<T>(0);  // dot

        T w = schurMatrixT.t<T>(i, i) - p;

        if (_Vals.t<T>(i, 1) < T(0)) {
          lastw = w;
          lastra = ra;
          lastsa = sa;
        } else {
          l = i;

          if (_Vals.t<T>(i, 1) == T(0)) {
            EigenValsAndVecs<T>::divideComplexNums(-ra, -sa, w, q, schurMatrixT.r<T>(i, n - 1),
                                                   schurMatrixT.r<T>(i, n));
          } else {
            T x = schurMatrixT.t<T>(i, i + 1);
            T y = schurMatrixT.t<T>(i + 1, i);
            T vr = (_Vals.t<T>(i, 0) - p) * (_Vals.t<T>(i, 0) - p) + _Vals.t<T>(i, 1) * _Vals.t<T>(i, 1) - q * q;
            T vi = (_Vals.t<T>(i, 0) - p) * T(2) * q;

            if ((vr == T(0)) && (vi == T(0)))
              vr = DataTypeUtils::eps<T>() * norm *
                   (math::sd_abs<T,T>(w) + math::sd_abs<T,T>(q) + math::sd_abs<T,T>(x) + math::sd_abs<T,T>(y) +
                    math::sd_abs<T,T>(lastw));

            EigenValsAndVecs<T>::divideComplexNums(x * lastra - lastw * ra + q * sa, x * lastsa - lastw * sa - q * ra,
                                                   vr, vi, schurMatrixT.r<T>(i, n - 1), schurMatrixT.r<T>(i, n));

            if (math::sd_abs<T,T>(x) > (math::sd_abs<T,T>(lastw) + math::sd_abs<T,T>(q))) {
              schurMatrixT.r<T>(i + 1, n - 1) =
                  (-ra - w * schurMatrixT.t<T>(i, n - 1) + q * schurMatrixT.t<T>(i, n)) / x;
              schurMatrixT.r<T>(i + 1, n) = (-sa - w * schurMatrixT.t<T>(i, n) - q * schurMatrixT.t<T>(i, n - 1)) / x;
            } else
              EigenValsAndVecs<T>::divideComplexNums(-lastra - y * schurMatrixT.t<T>(i, n - 1),
                                                     -lastsa - y * schurMatrixT.t<T>(i, n), lastw, q,
                                                     schurMatrixT.r<T>(i + 1, n - 1), schurMatrixT.r<T>(i + 1, n));
          }

          T t = math::sd_max<T>(math::sd_abs<T,T>(schurMatrixT.t<T>(i, n - 1)), math::sd_abs<T,T>(schurMatrixT.t<T>(i, n)));
          if ((DataTypeUtils::eps<T>() * t) * t > T(1)) schurMatrixT({i, numOfCols, n - 1, n + 1}) /= t;
        }
      }
      n--;
    } else
      THROW_EXCEPTION("ops::helpers::EigenValsAndVecs::calcEigenVecs: internal bug !");
  }

  for (int j = numOfCols - 1; j >= 0; j--) {
    NDArray assign = mmul(schurMatrixU({0, 0, 0, j + 1}, true), schurMatrixT({0, j + 1, j, j + 1}, true));
    schurMatrixU({0, 0, j, j + 1}, true).assign(&assign);
  }
}

template <typename T>
void EigenValsAndVecs<T>::calcPseudoEigenVecs(NDArray& schurMatrixT, NDArray& schurMatrixU) {
  calcPseudoEigenVecs_<T>(schurMatrixT, schurMatrixU, _Vals);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void calcEigenVecs_(NDArray& schurMatrixU, NDArray& _Vals, NDArray& _Vecs) {
  const T precision = T(2) * DataTypeUtils::eps<T>();

  const int numOfCols = schurMatrixU.sizeAt(1);

  for (int j = 0; j < numOfCols; ++j) {
    if (math::sd_abs<T,T>(_Vals.t<T>(j, 1)) <= math::sd_abs<T,T>(_Vals.t<T>(j, 0)) * precision ||
        j + 1 == numOfCols) {  // real

      _Vecs.syncToDevice();
      NDArray assign = schurMatrixU({0, 0, j, j + 1});
      _Vecs({0, 0, j, j + 1, 0, 1}).assign(&assign);
      _Vecs({0, 0, j, j + 1, 1, 2}) = (T)0;

      // normalize
      const T norm2 = _Vecs({0, 0, j, j + 1, 0, 1}).reduceNumber(reduce::SquaredNorm).template t<T>(0);
      if (norm2 > (T)0) _Vecs({0, 0, j, j + 1, 0, 1}) /= math::sd_sqrt<T, T>(norm2);
    } else {  // complex

      for (int i = 0; i < numOfCols; ++i) {
        _Vecs.r<T>(i, j, 0) = _Vecs.r<T>(i, j + 1, 0) = schurMatrixU.t<T>(i, j);
        _Vecs.r<T>(i, j, 1) = schurMatrixU.t<T>(i, j + 1);
        _Vecs.r<T>(i, j + 1, 1) = -schurMatrixU.t<T>(i, j + 1);
      }

      // normalize
      T norm2 = _Vecs({0, 0, j, j + 1, 0, 0}).reduceNumber(reduce::SquaredNorm).template t<T>(0);
      if (norm2 > (T)0) _Vecs({0, 0, j, j + 1, 0, 0}) /= math::sd_sqrt<T, T>(norm2);

      // normalize
      norm2 = _Vecs({0, 0, j + 1, j + 2, 0, 0}).reduceNumber(reduce::SquaredNorm).template t<T>(0);
      if (norm2 > (T)0) _Vecs({0, 0, j + 1, j + 2, 0, 0}) /= math::sd_sqrt<T, T>(norm2);

      ++j;
    }
  }
}

template <typename T>
void EigenValsAndVecs<T>::calcEigenVecs(NDArray& schurMatrixU) {
  calcEigenVecs_<T>(schurMatrixU, _Vals, _Vecs);
}

template <typename T>
void eig_(NDArray& input, NDArray& vals, NDArray& vecs) {
  assert(input.rankOf() == 2 && "input is not a matrix");
  assert(input.sizeAt(0) == input.sizeAt(1) && "input is not a square matrix");
  assert(vals.rankOf() == 2 && vals.sizeAt(0) == input.sizeAt(0) && vals.sizeAt(1) == 2 &&
         "incorrect shape for the eigenvalue results vals");
  assert(vecs.rankOf() == 3 && vecs.sizeAt(0) == input.sizeAt(0) && vecs.sizeAt(1) == input.sizeAt(0) &&
         vecs.sizeAt(2) == 2 && "incorrect shape for the eigenvector results vecs");

  Schur<T> schur(input);
  NDArray& schurMatrixU = schur.u;
  NDArray& schurMatrixT = schur.t;
  calcEigenVals_<T>(schurMatrixT, vals);
  calcPseudoEigenVecs_<T>(schurMatrixT, schurMatrixU, vals);
  calcEigenVecs_<T>(schurMatrixU, vals, vecs);
}

void eig(NDArray& input, NDArray& vals, NDArray& vecs) {
  BUILD_SINGLE_SELECTOR(input.dataType(), eig_, (input, vals, vecs), SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE(template void eig_, (NDArray& input, NDArray& vals, NDArray& vecs), SD_FLOAT_TYPES);

BUILD_SINGLE_TEMPLATE(template class EigenValsAndVecs, , SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
