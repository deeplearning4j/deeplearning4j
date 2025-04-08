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
//
#include <array/ResultSet.h>
#include <graph/FlatUtils.h>

namespace sd {
ResultSet::ResultSet() {
  //
}

ResultSet::ResultSet(const ::graph::FlatResult* result) {
  for (size_t e = 0; e < result->variables()->size(); e++) {
    auto var = result->variables()->Get(e);

    NDArray* array;

    if (var->ndarray() != nullptr) {
      array = graph::FlatUtils::fromFlatArray(var->ndarray());
    } else if (var->shape() != nullptr) {
      std::vector<LongType> shapeInfo;
      for (size_t i = 0; i < var->shape()->size(); i++) {
        shapeInfo.emplace_back(var->shape()->Get(i));
      }

      // we just create empty array here
      int s0 = shapeInfo.at(0);

      std::vector<LongType> shape;
      for (int i = 0; i < s0; i++) {
        shape.emplace_back(shapeInfo.at(i + 1));
      }

      array =
          new NDArray((char)shapeInfo.at(shapeInfo.size() - 1), shape, DataTypeUtils::fromFlatDataType(var->dtype()));
    } else {
      sd_printf("Either shape or NDArray should be defined in FlatResult variable\n", "");
      THROW_EXCEPTION("Empty variable");
    }

    _content.push_back(array);
  }
}

ResultSet::ResultSet(const ResultSet& other) noexcept {
  for (const auto v : other._content) _content.emplace_back(v);

  _status = other._status;
  _removable = false;
}

////////////////////////////////////////////////////////////////////////
// move constructor
ResultSet::ResultSet(ResultSet&& other) noexcept {
  _content = std::move(other._content);
  _status = other._status;
  _removable = other._removable;
  other._removable = false;
}

////////////////////////////////////////////////////////////////////////
// move assignment operator
ResultSet& ResultSet::operator=(ResultSet&& other) noexcept {
  if (this == &other) return *this;

  delContent();

  _content = std::move(other._content);

  _status = other._status;
  _removable = other._removable;
  other._removable = false;

  return *this;
}

ResultSet& ResultSet::operator=(const ResultSet& other) noexcept {
  if (this == &other) return *this;

  delContent();

  for (const auto v : other._content) _content.push_back(v);

  _status = other._status;
  _removable = false;

  return *this;
}

void ResultSet::delContent() {
  if (_removable) {
    std::vector<DataBuffer *> deleted;
    for (auto v : _content) {
      auto buffer = v->dataBuffer();
      deleted.push_back(buffer);
      if (!v->isView() && v->shapeInfo() != nullptr) {
        delete v;
      }
    }
  }
}

ResultSet::~ResultSet() { delContent(); }

void ResultSet::printIndexedBuffers() {
  for (size_t e = 0; e < _content.size(); e++) {
    auto array = _content.at(e);
    auto strVal = "Array e: " + std::to_string(e) + " is: ";
    array->printIndexedBuffer(strVal.c_str());
  }

}
void ResultSet::setNonRemovable() { _removable = false; }

int ResultSet::size() { return (int)_content.size(); }

NDArray* ResultSet::at(const unsigned long idx) const { return _content.at(idx); }

NDArray* ResultSet::operator[](const unsigned long idx) const { return _content[idx]; }

void ResultSet::push_back(NDArray* array) { _content.emplace_back(array); }

Status ResultSet::status() { return _status; }

void ResultSet::setStatus(Status status) { _status = status; }

void ResultSet::purge() { _content.clear(); }
}  // namespace sd
