/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */


#ifndef FLATBUFFERS_GENERATED_REQUEST_SD_GRAPH_H_
#define FLATBUFFERS_GENERATED_REQUEST_SD_GRAPH_H_
#include "flatbuffers/flatbuffers.h"
#include "array_generated.h"
#include "config_generated.h"
#include "utils_generated.h"
#include "variable_generated.h"

namespace sd {
namespace graph {

struct FlatInferenceRequest;

struct FlatInferenceRequest FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_ID = 4,
    VT_VARIABLES = 6,
    VT_CONFIGURATION = 8
  };
  int64_t id() const {
    return GetField<int64_t>(VT_ID, 0);
  }
  const flatbuffers::Vector<flatbuffers::Offset<FlatVariable>> *variables() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<FlatVariable>> *>(VT_VARIABLES);
  }
  const FlatConfiguration *configuration() const {
    return GetPointer<const FlatConfiguration *>(VT_CONFIGURATION);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<int64_t>(verifier, VT_ID) &&
           VerifyOffset(verifier, VT_VARIABLES) &&
           verifier.VerifyVector(variables()) &&
           verifier.VerifyVectorOfTables(variables()) &&
           VerifyOffset(verifier, VT_CONFIGURATION) &&
           verifier.VerifyTable(configuration()) &&
           verifier.EndTable();
  }
};

struct FlatInferenceRequestBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_id(int64_t id) {
    fbb_.AddElement<int64_t>(FlatInferenceRequest::VT_ID, id, 0);
  }
  void add_variables(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<FlatVariable>>> variables) {
    fbb_.AddOffset(FlatInferenceRequest::VT_VARIABLES, variables);
  }
  void add_configuration(flatbuffers::Offset<FlatConfiguration> configuration) {
    fbb_.AddOffset(FlatInferenceRequest::VT_CONFIGURATION, configuration);
  }
  explicit FlatInferenceRequestBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  FlatInferenceRequestBuilder &operator=(const FlatInferenceRequestBuilder &);
  flatbuffers::Offset<FlatInferenceRequest> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FlatInferenceRequest>(end);
    return o;
  }
};

inline flatbuffers::Offset<FlatInferenceRequest> CreateFlatInferenceRequest(
    flatbuffers::FlatBufferBuilder &_fbb,
    int64_t id = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<FlatVariable>>> variables = 0,
    flatbuffers::Offset<FlatConfiguration> configuration = 0) {
  FlatInferenceRequestBuilder builder_(_fbb);
  builder_.add_id(id);
  builder_.add_configuration(configuration);
  builder_.add_variables(variables);
  return builder_.Finish();
}

inline flatbuffers::Offset<FlatInferenceRequest> CreateFlatInferenceRequestDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    int64_t id = 0,
    const std::vector<flatbuffers::Offset<FlatVariable>> *variables = nullptr,
    flatbuffers::Offset<FlatConfiguration> configuration = 0) {
  auto variables__ = variables ? _fbb.CreateVector<flatbuffers::Offset<FlatVariable>>(*variables) : 0;
  return CreateFlatInferenceRequest(
      _fbb,
      id,
      variables__,
      configuration);
}

inline const FlatInferenceRequest *GetFlatInferenceRequest(const void *buf) {
  return flatbuffers::GetRoot<FlatInferenceRequest>(buf);
}

inline const FlatInferenceRequest *GetSizePrefixedFlatInferenceRequest(const void *buf) {
  return flatbuffers::GetSizePrefixedRoot<FlatInferenceRequest>(buf);
}

inline bool VerifyFlatInferenceRequestBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifyBuffer<FlatInferenceRequest>(nullptr);
}

inline bool VerifySizePrefixedFlatInferenceRequestBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifySizePrefixedBuffer<FlatInferenceRequest>(nullptr);
}

inline void FinishFlatInferenceRequestBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<FlatInferenceRequest> root) {
  fbb.Finish(root);
}

inline void FinishSizePrefixedFlatInferenceRequestBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<FlatInferenceRequest> root) {
  fbb.FinishSizePrefixed(root);
}

}  // namespace graph
}  // namespace sd

#endif  // FLATBUFFERS_GENERATED_REQUEST_SD_GRAPH_H_
