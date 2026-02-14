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
// @author Yurii Shyrma (iuriish@yahoo.com)
//
#include <execution/Threads.h>
#include <helpers/ConstantTadHelper.h>
#include <ops/declarable/helpers/lrn.h>
#if NOT_EXCLUDED(OP_lrn)
namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static sd::Status lrnFunctor_(sd::graph::Context& block, NDArray* input, NDArray* output, int depth, float bias,
                              float alpha, float beta) {
  sd_debug("MKL-DNN is not used for lrn!\n", 0);

  const int rank = input->rankOf();

  auto inTadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(),
                                                                             rank - 1);
  std::shared_ptr<TadPack> outTadPack;

  if (shape::haveSameShapeAndStrides(input->shapeInfo(), output->shapeInfo()))
    outTadPack = inTadPack;
  else
    outTadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(),
                                                                       rank - 1);

  const sd::LongType numOfTads = inTadPack->numberOfTads();
  const sd::LongType tadLen = input->sizeAt(-1);

  const sd::LongType* inTadOffsets = inTadPack->primaryOffsets();
  const sd::LongType* outTadOffsets = outTadPack->primaryOffsets();

  const sd::LongType inTadRank = shape::rank(inTadPack->primaryShapeInfo());
  const sd::LongType outTadRank = shape::rank(outTadPack->primaryShapeInfo());
  const sd::LongType* inTadShape = shape::shapeOf(inTadPack->primaryShapeInfo());
  const sd::LongType* outTadShape = shape::shapeOf(outTadPack->primaryShapeInfo());
  const sd::LongType* inTadStride = shape::stride(inTadPack->primaryShapeInfo());
  const sd::LongType* outTadStride = shape::stride(outTadPack->primaryShapeInfo());

  const T* inBuff = reinterpret_cast<T*>(input->buffer());
  T* outBuff = reinterpret_cast<T*>(output->buffer());

  const T tbias = static_cast<T>(bias);
  const T tbeta = static_cast<T>(beta);

  auto func = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; i++) {
      const T* x = inBuff + inTadOffsets[i];
      T* y = outBuff + outTadOffsets[i];

      T prev = static_cast<T>(0);

      // calculate squared sum of elements per each j-th element range [j - depth, j + depth + 1]
      // we store each squared sum in corresponding element of y array
      for (sd::LongType j = 0; j < tadLen; ++j) {
        const sd::LongType begin = sd::math::sd_max<int>(0, j - depth);
        const sd::LongType last = depth + j + 1;
        const sd::LongType end = sd::math::sd_min<int>(last, tadLen);

        sd::LongType coords[SD_MAX_RANK];
        sd::LongType inOff, outOff;
        INDEX2COORDS(j, inTadRank, inTadShape, coords);
        COORDS2INDEX(inTadRank, inTadStride, coords, inOff);
        COORDS2INDEX(outTadRank, outTadStride, coords, outOff);

        if (j == 0) {
          for (sd::LongType s = begin; s < end; ++s) {
            sd::LongType sCoords[SD_MAX_RANK];
            sd::LongType sOff;
            INDEX2COORDS(s, inTadRank, inTadShape, sCoords);
            COORDS2INDEX(inTadRank, inTadStride, sCoords, sOff);
            prev = prev + x[sOff] * x[sOff];
          }
          y[outOff] = prev;
        } else if (begin == 0 && last <= tadLen) {
          sd::LongType eCoords[SD_MAX_RANK];
          sd::LongType eOff;
          INDEX2COORDS(end - 1, inTadRank, inTadShape, eCoords);
          COORDS2INDEX(inTadRank, inTadStride, eCoords, eOff);
          y[outOff] = prev + x[eOff] * x[eOff];
        } else if (begin > 0 && last <= tadLen) {
          sd::LongType eCoords[SD_MAX_RANK], bCoords[SD_MAX_RANK];
          sd::LongType eOff, bOff;
          INDEX2COORDS(end - 1, inTadRank, inTadShape, eCoords);
          COORDS2INDEX(inTadRank, inTadStride, eCoords, eOff);
          INDEX2COORDS(begin - 1, inTadRank, inTadShape, bCoords);
          COORDS2INDEX(inTadRank, inTadStride, bCoords, bOff);
          y[outOff] = prev + x[eOff] * x[eOff] - x[bOff] * x[bOff];
        } else if (begin > 0 && last > tadLen) {
          sd::LongType bCoords[SD_MAX_RANK];
          sd::LongType bOff;
          INDEX2COORDS(begin - 1, inTadRank, inTadShape, bCoords);
          COORDS2INDEX(inTadRank, inTadStride, bCoords, bOff);
          y[outOff] = prev - x[bOff] * x[bOff];
        } else {
          y[outOff] = prev;
        }

        if (j != 0) prev = y[outOff];

        y[outOff] = x[inOff] / sd::math::sd_pow<T, T, T>(tbias + alpha * prev, tbeta);
      }
    }
  };

  samediff::Threads::parallel_tad(func, 0, numOfTads);
  return sd::Status::OK;
}

BUILD_SINGLE_TEMPLATE( sd::Status lrnFunctor_,
                      (sd::graph::Context & block, NDArray* input, NDArray* output, int depth, float bias, float alpha,
                       float beta),
                      SD_FLOAT_TYPES);

sd::Status lrnFunctor(sd::graph::Context& block, NDArray* input, NDArray* output, int depth, double bias, double alpha,
                      double beta) {
  BUILD_SINGLE_SELECTOR(input->dataType(), return lrnFunctor_, (block, input, output, depth, bias, alpha, beta),
                        SD_FLOAT_TYPES);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void lrnBP_(NDArray& input, NDArray& gradO, NDArray& gradI, const int depth, const float bias,
                   const float alpha, const float beta) {
  const int rank = input.rankOf();

  auto inTadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(input.shapeInfo(),
                                                                             rank - 1);
  std::shared_ptr<TadPack> gradITadPack;

  if (shape::haveSameShapeAndStrides(input.shapeInfo(), gradI.shapeInfo()))
    gradITadPack = inTadPack;
  else
    gradITadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(gradI.shapeInfo(),
                                                                         rank - 1);

  const sd::LongType numOfTads = inTadPack->numberOfTads();
  const sd::LongType tadLen = input.sizeAt(-1);

  const sd::LongType* inTadOffsets = inTadPack->primaryOffsets();
  const sd::LongType* gradITadOffsets = gradITadPack->primaryOffsets();

  const sd::LongType inTadRank = shape::rank(inTadPack->primaryShapeInfo());
  const sd::LongType gradITadRank = shape::rank(gradITadPack->primaryShapeInfo());
  const sd::LongType* inTadShape = shape::shapeOf(inTadPack->primaryShapeInfo());
  const sd::LongType* gradITadShape = shape::shapeOf(gradITadPack->primaryShapeInfo());
  const sd::LongType* inTadStride = shape::stride(inTadPack->primaryShapeInfo());
  const sd::LongType* gradITadStride = shape::stride(gradITadPack->primaryShapeInfo());

  const X* inBuff = reinterpret_cast<X const*>(input.buffer());
  Y* gradIBuff = reinterpret_cast<Y*>(gradI.buffer());

  const Y tbias = static_cast<Y>(bias);
  const Y tbeta = static_cast<Y>(beta);
  const Y talpha = static_cast<Y>(alpha);
  const Y coeff = talpha * tbeta;

  auto func = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; i++) {
      const X* x = inBuff + inTadOffsets[i];
      Y* y = gradIBuff + gradITadOffsets[i];

      // this loop calculates squared sum of elements per each j-th element range [j - depth, j + depth + 1]
      // we store each squared sum in corresponding element of y array
      for (sd::LongType j = 0; j < tadLen; ++j) {
        const sd::LongType begin = sd::math::sd_max<int>(0, j - depth);
        const sd::LongType last = depth + j + 1;
        const sd::LongType end = sd::math::sd_min<int>(last, tadLen);

        sd::LongType jInCoords[SD_MAX_RANK], jGradCoords[SD_MAX_RANK];
        sd::LongType jInOff, jGradOff;
        INDEX2COORDS(j, inTadRank, inTadShape, jInCoords);
        COORDS2INDEX(inTadRank, inTadStride, jInCoords, jInOff);
        INDEX2COORDS(j, gradITadRank, gradITadShape, jGradCoords);
        COORDS2INDEX(gradITadRank, gradITadStride, jGradCoords, jGradOff);

        if (j == 0) {
          y[jGradOff] = 0;
          for (sd::LongType s = begin; s < end; ++s) {
            sd::LongType sCoords[SD_MAX_RANK];
            sd::LongType sOff;
            INDEX2COORDS(s, inTadRank, inTadShape, sCoords);
            COORDS2INDEX(inTadRank, inTadStride, sCoords, sOff);
            y[jGradOff] = y[jGradOff] + x[sOff] * x[sOff];
          }
        } else {
          sd::LongType jm1GradCoords[SD_MAX_RANK];
          sd::LongType jm1GradOff;
          INDEX2COORDS(j - 1, gradITadRank, gradITadShape, jm1GradCoords);
          COORDS2INDEX(gradITadRank, gradITadStride, jm1GradCoords, jm1GradOff);

          if (begin == 0 && last <= tadLen) {
            sd::LongType eCoords[SD_MAX_RANK];
            sd::LongType eOff;
            INDEX2COORDS(end - 1, inTadRank, inTadShape, eCoords);
            COORDS2INDEX(inTadRank, inTadStride, eCoords, eOff);
            y[jGradOff] = y[jm1GradOff] + x[eOff] * x[eOff];
          } else if (begin > 0 && last <= tadLen) {
            sd::LongType eCoords[SD_MAX_RANK], bCoords[SD_MAX_RANK];
            sd::LongType eOff, bOff;
            INDEX2COORDS(end - 1, inTadRank, inTadShape, eCoords);
            COORDS2INDEX(inTadRank, inTadStride, eCoords, eOff);
            INDEX2COORDS(begin - 1, inTadRank, inTadShape, bCoords);
            COORDS2INDEX(inTadRank, inTadStride, bCoords, bOff);
            y[jGradOff] = y[jm1GradOff] + x[eOff] * x[eOff] - x[bOff] * x[bOff];
          } else if (begin > 0 && last > tadLen) {
            sd::LongType bCoords[SD_MAX_RANK];
            sd::LongType bOff;
            INDEX2COORDS(begin - 1, inTadRank, inTadShape, bCoords);
            COORDS2INDEX(inTadRank, inTadStride, bCoords, bOff);
            y[jGradOff] = y[jm1GradOff] - x[bOff] * x[bOff];
          } else {
            y[jGradOff] = y[jm1GradOff];
          }
        }
      }

      Y* factor = nullptr;
      ALLOCATE(factor, nullptr, tadLen, Y);

      Y prev = static_cast<Y>(0);
      // second loop calculates derivatives using information gained in first loop above
      for (sd::LongType j = 0; j < tadLen; ++j) {
        const sd::LongType begin = sd::math::sd_max<int>(0, j - depth);
        const sd::LongType last = depth + j + 1;
        const sd::LongType end = sd::math::sd_min<int>(last, tadLen);

        sd::LongType jInCoords[SD_MAX_RANK], jGradCoords[SD_MAX_RANK];
        sd::LongType jInOff, jGradOff;
        INDEX2COORDS(j, inTadRank, inTadShape, jInCoords);
        COORDS2INDEX(inTadRank, inTadStride, jInCoords, jInOff);
        INDEX2COORDS(j, gradITadRank, gradITadShape, jGradCoords);
        COORDS2INDEX(gradITadRank, gradITadStride, jGradCoords, jGradOff);

        Y init = tbias + talpha * y[jGradOff];

        if (j == 0) {
          for (sd::LongType s = begin; s < end; ++s) {
            sd::LongType sInCoords[SD_MAX_RANK], sGradCoords[SD_MAX_RANK];
            sd::LongType sInOff, sGradOff;
            INDEX2COORDS(s, inTadRank, inTadShape, sInCoords);
            COORDS2INDEX(inTadRank, inTadStride, sInCoords, sInOff);
            INDEX2COORDS(s, gradITadRank, gradITadShape, sGradCoords);
            COORDS2INDEX(gradITadRank, gradITadStride, sGradCoords, sGradOff);
            factor[s] = sd::math::sd_pow<Y, Y, Y>(tbias + talpha * y[sGradOff], -tbeta - 1);
            prev = prev + x[sInOff] * factor[s];
          }
          y[jGradOff] = prev;
        } else if (begin == 0 && last <= tadLen) {
          sd::LongType eInCoords[SD_MAX_RANK], eGradCoords[SD_MAX_RANK];
          sd::LongType eInOff, eGradOff;
          INDEX2COORDS(end - 1, inTadRank, inTadShape, eInCoords);
          COORDS2INDEX(inTadRank, inTadStride, eInCoords, eInOff);
          INDEX2COORDS(end - 1, gradITadRank, gradITadShape, eGradCoords);
          COORDS2INDEX(gradITadRank, gradITadStride, eGradCoords, eGradOff);
          factor[end - 1] = sd::math::sd_pow<Y, Y, Y>(tbias + talpha * y[eGradOff], -tbeta - 1);
          y[jGradOff] = prev + x[eInOff] * factor[end - 1];
        } else if (begin > 0 && last <= tadLen) {
          sd::LongType eInCoords[SD_MAX_RANK], eGradCoords[SD_MAX_RANK];
          sd::LongType eInOff, eGradOff;
          INDEX2COORDS(end - 1, inTadRank, inTadShape, eInCoords);
          COORDS2INDEX(inTadRank, inTadStride, eInCoords, eInOff);
          INDEX2COORDS(end - 1, gradITadRank, gradITadShape, eGradCoords);
          COORDS2INDEX(gradITadRank, gradITadStride, eGradCoords, eGradOff);
          sd::LongType bInCoords[SD_MAX_RANK];
          sd::LongType bInOff;
          INDEX2COORDS(begin - 1, inTadRank, inTadShape, bInCoords);
          COORDS2INDEX(inTadRank, inTadStride, bInCoords, bInOff);
          factor[end - 1] = sd::math::sd_pow<Y, Y, Y>(tbias + talpha * y[eGradOff], -tbeta - 1);
          y[jGradOff] = prev + x[eInOff] * factor[end - 1] - x[bInOff] * factor[begin - 1];
        } else if (begin > 0 && last > tadLen) {
          sd::LongType bInCoords[SD_MAX_RANK];
          sd::LongType bInOff;
          INDEX2COORDS(begin - 1, inTadRank, inTadShape, bInCoords);
          COORDS2INDEX(inTadRank, inTadStride, bInCoords, bInOff);
          y[jGradOff] = prev - x[bInOff] * factor[begin - 1];
        } else {
          y[jGradOff] = prev;
        }

        if (j != 0) prev = y[jGradOff];

        y[jGradOff] = factor[j] * init - 2 * x[jInOff] * coeff * prev;
      }

      RELEASE(factor, nullptr);
    }
  };

  samediff::Threads::parallel_tad(func, 0, numOfTads);
  gradI *= gradO;
}

void lrnBP(sd::graph::Context& block, NDArray& input, NDArray& gradO, NDArray& gradI, const int depth,
           const float bias, const float alpha, const float beta) {
  BUILD_DOUBLE_SELECTOR(input.dataType(), gradO.dataType(), lrnBP_, (input, gradO, gradI, depth, bias, alpha, beta),
                        SD_FLOAT_TYPES, SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif