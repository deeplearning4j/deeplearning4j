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


#ifndef FLATBUFFERS_GENERATED_VARIABLE_SD_GRAPH_H_
#define FLATBUFFERS_GENERATED_VARIABLE_SD_GRAPH_H_

#include "flatbuffers/flatbuffers.h"

#include "array_generated.h"
#include "utils_generated.h"

namespace sd {
namespace graph {

struct FlatVariable;
struct FlatVariableBuilder;

enum VarType {
  VarType_VARIABLE = 0,
  VarType_CONSTANT = 1,
  VarType_ARRAY = 2,
  VarType_PLACEHOLDER = 3,
  VarType_MIN = VarType_VARIABLE,
  VarType_MAX = VarType_PLACEHOLDER
};

inline const VarType (&EnumValuesVarType())[4] {
  static const VarType values[] = {
    VarType_VARIABLE,
    VarType_CONSTANT,
    VarType_ARRAY,
    VarType_PLACEHOLDER
  };
  return values;
}

inline const char * const *EnumNamesVarType() {
  static const char * const names[5] = {
    "VARIABLE",
    "CONSTANT",
    "ARRAY",
    "PLACEHOLDER",
    nullptr
  };
  return names;
}

inline const char *EnumNameVarType(VarType e) {
  if (flatbuffers::IsOutRange(e, VarType_VARIABLE, VarType_PLACEHOLDER)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesVarType()[index];
}

struct FlatVariable FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FlatVariableBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_ID = 4,
    VT_NAME = 6,
    VT_DTYPE = 8,
    VT_SHAPE = 10,
    VT_NDARRAY = 12,
    VT_DEVICE = 14,
    VT_VARIABLETYPE = 16,
    VT_CONTROLDEPS = 18,
    VT_CONTROLDEPFOROP = 20,
    VT_CONTROLDEPSFORVAR = 22
  };
  const sd::graph::IntPair *id() const {
    return GetPointer<const sd::graph::IntPair *>(VT_ID);
  }
  const flatbuffers::String *name() const {
    return GetPointer<const flatbuffers::String *>(VT_NAME);
  }
  sd::graph::DType dtype() const {
    return static_cast<sd::graph::DType>(GetField<int8_t>(VT_DTYPE, 0));
  }
  const flatbuffers::Vector<int64_t> *shape() const {
    return GetPointer<const flatbuffers::Vector<int64_t> *>(VT_SHAPE);
  }
  const sd::graph::FlatArray *ndarray() const {
    return GetPointer<const sd::graph::FlatArray *>(VT_NDARRAY);
  }
  int32_t device() const {
    return GetField<int32_t>(VT_DEVICE, 0);
  }
  sd::graph::VarType variabletype() const {
    return static_cast<sd::graph::VarType>(GetField<int8_t>(VT_VARIABLETYPE, 0));
  }
  const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *controlDeps() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *>(VT_CONTROLDEPS);
  }
  const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *controlDepForOp() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *>(VT_CONTROLDEPFOROP);
  }
  const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *controlDepsForVar() const {
    return GetPointer<const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *>(VT_CONTROLDEPSFORVAR);
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_ID) &&
           verifier.VerifyTable(id()) &&
           VerifyOffset(verifier, VT_NAME) &&
           verifier.VerifyString(name()) &&
           VerifyField<int8_t>(verifier, VT_DTYPE) &&
           VerifyOffset(verifier, VT_SHAPE) &&
           verifier.VerifyVector(shape()) &&
           VerifyOffset(verifier, VT_NDARRAY) &&
           verifier.VerifyTable(ndarray()) &&
           VerifyField<int32_t>(verifier, VT_DEVICE) &&
           VerifyField<int8_t>(verifier, VT_VARIABLETYPE) &&
           VerifyOffset(verifier, VT_CONTROLDEPS) &&
           verifier.VerifyVector(controlDeps()) &&
           verifier.VerifyVectorOfStrings(controlDeps()) &&
           VerifyOffset(verifier, VT_CONTROLDEPFOROP) &&
           verifier.VerifyVector(controlDepForOp()) &&
           verifier.VerifyVectorOfStrings(controlDepForOp()) &&
           VerifyOffset(verifier, VT_CONTROLDEPSFORVAR) &&
           verifier.VerifyVector(controlDepsForVar()) &&
           verifier.VerifyVectorOfStrings(controlDepsForVar()) &&
           verifier.EndTable();
  }
};

struct FlatVariableBuilder {
  typedef FlatVariable Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_id(flatbuffers::Offset<sd::graph::IntPair> id) {
    fbb_.AddOffset(FlatVariable::VT_ID, id);
  }
  void add_name(flatbuffers::Offset<flatbuffers::String> name) {
    fbb_.AddOffset(FlatVariable::VT_NAME, name);
  }
  void add_dtype(sd::graph::DType dtype) {
    fbb_.AddElement<int8_t>(FlatVariable::VT_DTYPE, static_cast<int8_t>(dtype), 0);
  }
  void add_shape(flatbuffers::Offset<flatbuffers::Vector<int64_t>> shape) {
    fbb_.AddOffset(FlatVariable::VT_SHAPE, shape);
  }
  void add_ndarray(flatbuffers::Offset<sd::graph::FlatArray> ndarray) {
    fbb_.AddOffset(FlatVariable::VT_NDARRAY, ndarray);
  }
  void add_device(int32_t device) {
    fbb_.AddElement<int32_t>(FlatVariable::VT_DEVICE, device, 0);
  }
  void add_variabletype(sd::graph::VarType variabletype) {
    fbb_.AddElement<int8_t>(FlatVariable::VT_VARIABLETYPE, static_cast<int8_t>(variabletype), 0);
  }
  void add_controlDeps(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> controlDeps) {
    fbb_.AddOffset(FlatVariable::VT_CONTROLDEPS, controlDeps);
  }
  void add_controlDepForOp(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> controlDepForOp) {
    fbb_.AddOffset(FlatVariable::VT_CONTROLDEPFOROP, controlDepForOp);
  }
  void add_controlDepsForVar(flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> controlDepsForVar) {
    fbb_.AddOffset(FlatVariable::VT_CONTROLDEPSFORVAR, controlDepsForVar);
  }
  explicit FlatVariableBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  FlatVariableBuilder &operator=(const FlatVariableBuilder &);
  flatbuffers::Offset<FlatVariable> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FlatVariable>(end);
    return o;
  }
};

inline flatbuffers::Offset<FlatVariable> CreateFlatVariable(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<sd::graph::IntPair> id = 0,
    flatbuffers::Offset<flatbuffers::String> name = 0,
    sd::graph::DType dtype = sd::graph::DType_INHERIT,
    flatbuffers::Offset<flatbuffers::Vector<int64_t>> shape = 0,
    flatbuffers::Offset<sd::graph::FlatArray> ndarray = 0,
    int32_t device = 0,
    sd::graph::VarType variabletype = sd::graph::VarType_VARIABLE,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> controlDeps = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> controlDepForOp = 0,
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> controlDepsForVar = 0) {
  FlatVariableBuilder builder_(_fbb);
  builder_.add_controlDepsForVar(controlDepsForVar);
  builder_.add_controlDepForOp(controlDepForOp);
  builder_.add_controlDeps(controlDeps);
  builder_.add_device(device);
  builder_.add_ndarray(ndarray);
  builder_.add_shape(shape);
  builder_.add_name(name);
  builder_.add_id(id);
  builder_.add_variabletype(variabletype);
  builder_.add_dtype(dtype);
  return builder_.Finish();
}

inline flatbuffers::Offset<FlatVariable> CreateFlatVariableDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<sd::graph::IntPair> id = 0,
    const char *name = nullptr,
    sd::graph::DType dtype = sd::graph::DType_INHERIT,
    const std::vector<int64_t> *shape = nullptr,
    flatbuffers::Offset<sd::graph::FlatArray> ndarray = 0,
    int32_t device = 0,
    sd::graph::VarType variabletype = sd::graph::VarType_VARIABLE,
    const std::vector<flatbuffers::Offset<flatbuffers::String>> *controlDeps = nullptr,
    const std::vector<flatbuffers::Offset<flatbuffers::String>> *controlDepForOp = nullptr,
    const std::vector<flatbuffers::Offset<flatbuffers::String>> *controlDepsForVar = nullptr) {
  auto name__ = name ? _fbb.CreateString(name) : 0;
  auto shape__ = shape ? _fbb.CreateVector<int64_t>(*shape) : 0;
  auto controlDeps__ = controlDeps ? _fbb.CreateVector<flatbuffers::Offset<flatbuffers::String>>(*controlDeps) : 0;
  auto controlDepForOp__ = controlDepForOp ? _fbb.CreateVector<flatbuffers::Offset<flatbuffers::String>>(*controlDepForOp) : 0;
  auto controlDepsForVar__ = controlDepsForVar ? _fbb.CreateVector<flatbuffers::Offset<flatbuffers::String>>(*controlDepsForVar) : 0;
  return sd::graph::CreateFlatVariable(
      _fbb,
      id,
      name__,
      dtype,
      shape__,
      ndarray,
      device,
      variabletype,
      controlDeps__,
      controlDepForOp__,
      controlDepsForVar__);
}

inline const sd::graph::FlatVariable *GetFlatVariable(const void *buf) {
  return flatbuffers::GetRoot<sd::graph::FlatVariable>(buf);
}

inline const sd::graph::FlatVariable *GetSizePrefixedFlatVariable(const void *buf) {
  return flatbuffers::GetSizePrefixedRoot<sd::graph::FlatVariable>(buf);
}

inline bool VerifyFlatVariableBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifyBuffer<sd::graph::FlatVariable>(nullptr);
}

inline bool VerifySizePrefixedFlatVariableBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifySizePrefixedBuffer<sd::graph::FlatVariable>(nullptr);
}

inline void FinishFlatVariableBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<sd::graph::FlatVariable> root) {
  fbb.Finish(root);
}

inline void FinishSizePrefixedFlatVariableBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<sd::graph::FlatVariable> root) {
  fbb.FinishSizePrefixed(root);
}

}  // namespace graph
}  // namespace sd

#endif  // FLATBUFFERS_GENERATED_VARIABLE_SD_GRAPH_H_
