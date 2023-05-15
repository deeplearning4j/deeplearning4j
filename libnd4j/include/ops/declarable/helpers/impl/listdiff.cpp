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

#include <system/op_boilerplate.h>

#if NOT_EXCLUDED(OP_listdiff)
#include <ops/declarable/helpers/listdiff.h>

#include <vector>
//#include <memory>

namespace sd {
namespace ops {
namespace helpers {
template <typename T>
static sd::LongType listDiffCount_(NDArray* values, NDArray* keep) {
  sd::LongType saved = 0L;
  for (sd::LongType e = 0; e < values->lengthOf(); e++) {
    auto v = values->e<double>(e);
    ExtraArguments extras({v, 0.0, 10.0});
    auto idx = keep->indexReduceNumber(indexreduce::FirstIndex, &extras);
    auto index = idx.e<sd::LongType>(0);
    if (index < 0) saved++;
  }
  return saved;
}

sd::LongType listDiffCount(sd::LaunchContext* context, NDArray* values, NDArray* keep) {
  auto xType = values->dataType();

  NDArray::preparePrimaryUse({}, {values, keep});

  BUILD_SINGLE_SELECTOR(xType, return listDiffCount_, (values, keep), SD_COMMON_TYPES);

  NDArray::registerPrimaryUse({}, {values, keep});
}

BUILD_SINGLE_TEMPLATE(template sd::LongType listDiffCount_, (NDArray * values, NDArray* keep);, SD_COMMON_TYPES);

template <typename T>
static sd::Status listDiffFunctor_(NDArray* values, NDArray* keep, NDArray* output1, NDArray* output2) {
  std::vector<T> saved;
  std::vector<sd::LongType> indices;

  for (sd::LongType e = 0; e < values->lengthOf(); e++) {
    auto v = values->e<double>(e);
    ExtraArguments extras({v, 0.0, 10.0});
    NDArray idxScalar = keep->indexReduceNumber(indexreduce::FirstIndex, &extras);
    sd::LongType idx = idxScalar.e<sd::LongType>(0);

    if (idx < 0) {
      saved.emplace_back(v);
      indices.emplace_back(e);
    }
  }

  if (saved.size() == 0) {
    sd_printf("ListDiff: search returned no results", "");
    throw std::invalid_argument("Op validation failed");
  } else {
    auto z0 = output1;  // OUTPUT_VARIABLE(0); //new NDArray<T>('c', {(int) saved.size()});
    auto z1 = output2;  // OUTPUT_VARIABLE(1); //new NDArray<T>('c', {(int) saved.size()});

    if (z0->lengthOf() != saved.size()) {
      sd_printf("ListDiff: output/actual size mismatch", "");
      throw std::invalid_argument("Op validation failed");
    }

    if (z1->lengthOf() != saved.size()) {
      sd_printf("ListDiff: output/actual indices size mismatch", "");
      throw std::invalid_argument("Op validation failed");
    }
    memcpy(z0->buffer(), saved.data(), saved.size() * sizeof(T));
    for (int e = 0; e < indices.size(); e++) {
      z1->p(e, indices[e]);
    }
  }
  return sd::Status::OK;
}

sd::Status listDiffFunctor(sd::LaunchContext* context, NDArray* values, NDArray* keep, NDArray* output1,
                           NDArray* output2) {
  auto xType = values->dataType();

  NDArray::preparePrimaryUse({output1, output2}, {values, keep});

  sd::Status result = sd::Status::OK;

  if (DataTypeUtils::isR(xType)) {
    BUILD_SINGLE_SELECTOR(xType, result = listDiffFunctor_, (values, keep, output1, output2), SD_FLOAT_TYPES);
  } else if (DataTypeUtils::isZ(xType)) {
    BUILD_SINGLE_SELECTOR(xType, result = listDiffFunctor_, (values, keep, output1, output2), SD_INTEGER_TYPES);
  } else {
    THROW_EXCEPTION("ListDiff: Only integer and floating point data types are supported");
  }

  NDArray::registerPrimaryUse({output1, output2}, {values, keep});

  return result;
}

BUILD_SINGLE_TEMPLATE(template sd::Status listDiffFunctor_,
                      (NDArray * values, NDArray* keep, NDArray* output1, NDArray* output2);
                      , SD_FLOAT_TYPES);
BUILD_SINGLE_TEMPLATE(template sd::Status listDiffFunctor_,
                      (NDArray * values, NDArray* keep, NDArray* output1, NDArray* output2);
                      , SD_INTEGER_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif