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

#include <array/NDArrayList.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/stack.h>

#include <iterator>
#if NOT_EXCLUDED(OP_stack)
namespace sd {
NDArrayList::NDArrayList(int height, bool expandable) {
  _expandable = expandable;
  _elements.store(0);
  _counter.store(0);
  _id.first = 0;
  _id.second = 0;
  _height = height;
   sd_debug("\nCreating NDArrayList\n","");
}

NDArrayList::~NDArrayList() {
  sd_debug("\nDeleting NDArrayList: [%i]\n", _chunks.size());
  for (auto const& v : _chunks) delete v.second;

  _chunks.clear();
}

NDArray* NDArrayList::read(int idx) { return new NDArray(readRaw(idx)->dup()); }

sd::DataType NDArrayList::dataType() { return _dtype; }

NDArray* NDArrayList::readRaw(int idx) {
  if (_chunks.count(idx) < 1) {
    sd_debug("Non-existent chunk requested: [%i]\n", idx);
    THROW_EXCEPTION("Bad index");
  }

  return _chunks[idx];
}


NDArray* NDArrayList::remove(int idx) {
  if(!isWritten(idx)) {
    sd_debug("Non-existent chunk requested: [%i]\n", idx);
    THROW_EXCEPTION("Bad index");
  }

  delete _chunks[idx];

  _elements--;
  return new NDArray(readRaw(idx)->dup());
}


sd::Status NDArrayList::write(int idx, NDArray* array) {
  if (_chunks.count(idx) == 0)
    _elements++;
  else {
    delete _chunks[idx];
  }

  // we store reference shape on first write
  if (_chunks.empty()) {
    _dtype = array->dataType();

    if (_shape.empty()) {
      // adding leading 1 to shape
      _shape.emplace_back(1);
      for (int e = 0; e < array->rankOf(); e++) _shape.emplace_back(array->sizeAt(e));
    } else {
      // if shape is inferred (say, from split_list)
      if (static_cast<size_t>(array->rankOf()) == _shape.size()) {
        // skipping first dim
        for (size_t e = 1; e < _shape.size(); e++) {
          if (_shape[e] != array->sizeAt(e))
            return Logger::logStatusMsg(Status::BAD_INPUT,
                                        "NDArrayList: all arrays must have same size along inner dimensions");
        }
      } else if (static_cast<size_t>(array->rankOf()) == _shape.size() - 1) {
        // case like 2d _shape, and 1D rows
        for (size_t e = 1; e < _shape.size(); e++)
          if (_shape[e] != array->sizeAt(e - 1))
            return Logger::logStatusMsg(Status::BAD_INPUT,
                                        "NDArrayList: all arrays must have same size along inner dimensions");
      } else
        return Logger::logStatusMsg(Status::BAD_INPUT,
                                    "NDArrayList: all arrays must have same size along inner dimensions");
    }
  } else {
    if (array->dataType() != _dtype)
      return Logger::logStatusMsg(Status::BAD_INPUT, "NDArrayList: all arrays must have same data type");

    // if shape is inferred (say, from split_list)
    if (static_cast<size_t>(array->rankOf()) == _shape.size()) {
      // skipping first dim
      for (size_t e = 1; e < _shape.size(); e++) {
        if (_shape[e] != array->sizeAt(e))
          return Logger::logStatusMsg(Status::BAD_INPUT,
                                      "NDArrayList: all arrays must have same size along inner dimensions");
      }
    } else if (static_cast<size_t>(array->rankOf()) == _shape.size() - 1) {
      // case like 2d _shape, and 1D rows
      for (size_t e = 1; e < _shape.size(); e++)
        if (_shape[e] != array->sizeAt(e - 1))
          return Logger::logStatusMsg(Status::BAD_INPUT,
                                      "NDArrayList: all arrays must have same size along inner dimensions");
    } else
      return Logger::logStatusMsg(Status::BAD_INPUT,
                                  "NDArrayList: all arrays must have same size along inner dimensions");
  }


  // storing reference
  _chunks[idx] = array;

  return Status::OK;
}

std::vector<sd::LongType>& NDArrayList::shape() { return _shape; }

int NDArrayList::counter() { return _counter++; }

void NDArrayList::unstack(NDArray* array, LongType axis) {
  _axis = axis;
  std::vector<sd::LongType> args({axis});
  auto newAxis = ShapeUtils::evalDimsToExclude(array->rankOf(),1, args.data());
  auto result = array->allTensorsAlongDimension(*newAxis);
  for (sd::LongType e = 0; e < result.size(); e++) {
    auto chunk = result.at(e);
    write(e, new NDArray(chunk->dup(array->ordering())));
  }

  delete newAxis;
}

NDArray* NDArrayList::stack() {
  int numElements = _elements.load();
  if(numElements < 1) {
    return  new NDArray(NDArrayFactory::empty<double>());

  }
  std::vector<NDArray*> inputs(numElements);
  for (int e = 0; e < numElements; e++) {
    if(!_chunks[e]->isEmpty())
      _chunks[e]->syncToDevice();
    inputs[e] = _chunks[e];
  }

  if(inputs[0] == nullptr) {
    THROW_EXCEPTION("First input element was a null ptr!");
  }

  auto inShapeInfo = inputs[0]->shapeInfo();
  int rank = shape::rank(inShapeInfo);
  NDArray* array = nullptr;

  if (shape::isEmptyConst(inShapeInfo)) {
    switch (rank) {
      case 0: {
        if (numElements == 1) {
          std::vector<sd::LongType> shape = {0};
          array = new NDArray(inputs[0]->ordering(), shape, ArrayOptions::dataType(inShapeInfo), inputs[0]->getContext());
        } else {
          std::vector<sd::LongType> shape =  {(sd::LongType)numElements, 0};
          array = new NDArray('c', shape, ArrayOptions::dataType(inShapeInfo),
                              inputs[0]->getContext());
        }
      }
    }
  } else {

    std::vector<sd::LongType> outShape(inShapeInfo + 1, inShapeInfo + 1 + rank);
    outShape.insert(outShape.begin(), (sd::LongType)numElements);
    array =
        new NDArray(shape::order(inShapeInfo), outShape, ArrayOptions::dataType(inShapeInfo), inputs[0]->getContext());
  }

  ops::helpers::stack(inputs[0]->getContext(), inputs, *array, 0);

  return array;
}

std::pair<int, int>& NDArrayList::id() { return _id; }

std::string& NDArrayList::name() { return _name; }

sd::LaunchContext* NDArrayList::context() { return _context; }

int NDArrayList::elements() { return _elements.load(); }

int NDArrayList::height() {
  return (int)_chunks.size();
}

bool NDArrayList::isWritten(int index) {
  if (_chunks.count(index) > 0)
    return true;
  else
    return false;
}

NDArray* NDArrayList::pick(std::initializer_list<LongType> indices) {
  std::vector<LongType> idcs(indices);
  return pick(idcs);
}

NDArray* NDArrayList::pick(std::vector<LongType>& indices) {
  std::vector<sd::LongType> shape(_shape);

  shape[_axis] = indices.size();
  // do we have to enforce C order here?
  auto array = new NDArray('c', shape, _chunks[0]->dataType(), _context);
  const sd::LongType *axis2 = const_cast<sd::LongType *>(&_axis);
  std::vector<sd::LongType> *axis = ShapeUtils::evalDimsToExclude(shape.size(),1, axis2);
  auto tads = array->allTensorsAlongDimension(*axis);
  int indicesSize = indices.size();

  if (tads.size() != indicesSize) THROW_EXCEPTION("Number of TADs should match number of indices");

  for (int e = 0; e < indicesSize; e++) tads.at(e)->assign(_chunks[indices[e]]);

  delete axis;
  return array;
}

NDArrayList* NDArrayList::clone() {
  auto list = new NDArrayList(_height, _expandable);
  list->_axis = _axis;
  list->_id.first = _id.first;
  list->_id.second = _id.second;
  list->_name = _name;
  list->_elements.store(_elements.load());

  for (auto const& v : _chunks) {
    list->_chunks[v.first] = new NDArray(v.second->dup());
  }

  return list;
}

bool NDArrayList::equals(NDArrayList& other) {
  if (_axis != other._axis) return false;

  if (_chunks.size() != other._chunks.size()) return false;

  for (auto const& v : _chunks) {
    if (other._chunks.count(v.first) == 0) return false;

    auto arrThis = _chunks[v.first];
    auto arrThat = other._chunks[v.first];

    if (!arrThis->equalsTo(arrThat)) return false;
  }

  return true;
}
}  // namespace sd
#endif