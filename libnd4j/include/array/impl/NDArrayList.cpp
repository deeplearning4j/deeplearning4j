/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
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


#include <iterator>
#include <array/NDArrayList.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    NDArrayList::NDArrayList(int height, bool expandable) {
        _expandable = expandable;
        _elements.store(0);
        _counter.store(0);
        _id.first = 0;
        _id.second = 0;
        _height = height;
        //nd4j_printf("\nCreating NDArrayList\n","");
    }

    NDArrayList::~NDArrayList() {
        //nd4j_printf("\nDeleting NDArrayList: [%i]\n", _chunks.size());
        for (auto const& v : _chunks)
            delete v.second;

        _chunks.clear();
    }

    NDArray* NDArrayList::read(int idx) {
        return readRaw(idx)->dup();
    }

    nd4j::DataType NDArrayList::dataType() {
        return _dtype;
    }

    NDArray* NDArrayList::readRaw(int idx) {
        if (_chunks.count(idx) < 1) {
            nd4j_printf("Non-existent chunk requested: [%i]\n", idx);
            throw std::invalid_argument("Bad index");
        }

        return _chunks[idx];
    }

    Nd4jStatus NDArrayList::write(int idx, NDArray* array) {
        if (_chunks.count(idx) == 0)
            _elements++;
        else {
            delete _chunks[idx];
        }


        // we store reference shape on first write
        if (_chunks.empty()) {
            _dtype = array->dataType();

            if (_shape.empty()) {
                //adding leading 1 to shape
                _shape.emplace_back(1);
                for (int e = 0; e < array->rankOf(); e++)
                    _shape.emplace_back(array->sizeAt(e));
            } else {
                // if shape is inferred (say, from split_list)
                if (array->rankOf() == _shape.size()) {
                    // skipping first dim
                    for (int e = 1; e < _shape.size(); e++) {
                        if (_shape[e] != array->sizeAt(e))
                            return Status::CODE(ND4J_STATUS_BAD_INPUT, "NDArrayList: all arrays must have same size along inner dimensions");
                    }
                } else if (array->rankOf() == _shape.size() - 1) {
                    // case like 2d _shape, and 1D rows
                    for (int e = 1; e < _shape.size(); e++)
                        if (_shape[e] != array->sizeAt(e - 1))
                            return Status::CODE(ND4J_STATUS_BAD_INPUT, "NDArrayList: all arrays must have same size along inner dimensions");
                } else
                    return Status::CODE(ND4J_STATUS_BAD_INPUT, "NDArrayList: all arrays must have same size along inner dimensions");

            }
        } else {
            if (array->dataType() != _dtype)
                return Status::CODE(ND4J_STATUS_BAD_INPUT, "NDArrayList: all arrays must have same data type");


            // if shape is inferred (say, from split_list)
            if (array->rankOf() == _shape.size()) {
                // skipping first dim
                for (int e = 1; e < _shape.size(); e++) {
                    if (_shape[e] != array->sizeAt(e))
                        return Status::CODE(ND4J_STATUS_BAD_INPUT, "NDArrayList: all arrays must have same size along inner dimensions");
                }
            } else if (array->rankOf() == _shape.size() - 1) {
                // case like 2d _shape, and 1D rows
                for (int e = 1; e < _shape.size(); e++)
                    if (_shape[e] != array->sizeAt(e - 1))
                        return Status::CODE(ND4J_STATUS_BAD_INPUT, "NDArrayList: all arrays must have same size along inner dimensions");
            } else
                return Status::CODE(ND4J_STATUS_BAD_INPUT, "NDArrayList: all arrays must have same size along inner dimensions");
        }
        
        //_elements++;

        // storing reference
        _chunks[idx] = array;

        return Status::OK();
    }

    std::vector<Nd4jLong>& NDArrayList::shape() {
        return _shape;
    }

    int NDArrayList::counter() {
        return _counter++;
    }

    void NDArrayList::unstack(NDArray* array, int axis) {
        _axis = axis;
        std::vector<int> args({axis});
        auto newAxis = ShapeUtils::evalDimsToExclude(array->rankOf(), args);
        auto result = array->allTensorsAlongDimension(newAxis);
        for (int e = 0; e < result->size(); e++) {
            auto chunk = result->at(e);//->dup(array->ordering());
            write(e, chunk->dup(array->ordering()));
        }
        delete result;
    }

    NDArray* NDArrayList::stack() {
        // FIXME: this is bad for perf, but ok as poc
        nd4j::ops::stack op;
        std::vector<NDArray*> inputs;
        std::vector<double> targs;
        std::vector<Nd4jLong> iargs({0});
        std::vector<bool> bargs;
        int numElements = _elements.load();

        for (int e = 0; e < numElements; e++) {
            _chunks[e]->syncToDevice();
            inputs.emplace_back(_chunks[e]);
        }

        iargs.push_back(_axis);

        auto result = op.execute(inputs, {}, {}, {});

        auto array = result->at(0)->dup();

        delete result;

        return array;
    }

    std::pair<int,int>& NDArrayList::id() {
        return _id;
    }

    std::string& NDArrayList::name() {
        return _name;
    }

    nd4j::LaunchContext * NDArrayList::context() {
        return _context;
    }

    int NDArrayList::elements() {
        return _elements.load();
    }

    int NDArrayList::height() {
        //if (_height != 0)
        //    return _height;
        //else
            return (int) _chunks.size();
    }

    bool NDArrayList::isWritten(int index) {
        if (_chunks.count(index) > 0)
            return true;
        else
            return false;
    }

    NDArray* NDArrayList::pick(std::initializer_list<int> indices) {
        std::vector<int> idcs(indices);
        return pick(idcs);
    }

    NDArray* NDArrayList::pick(std::vector<int> &indices) {
        std::vector<Nd4jLong> shape(_shape);

        //shape.insert(shape.begin() + _axis, indices.size());
        shape[_axis] = indices.size();
        // do we have to enforce C order here?
        auto array = new NDArray('c', shape, _chunks[0]->dataType(), _context);
        std::vector<int> axis = ShapeUtils::evalDimsToExclude(shape.size(), {_axis});
        auto tads = array->allTensorsAlongDimension(axis);
        int indicesSize = indices.size();

        if (tads->size() != indicesSize)
            throw std::runtime_error("Number of TADs should match number of indices");

        for (int e = 0; e < indicesSize; e++)
            tads->at(e)->assign(_chunks[indices[e]]);

        delete tads;

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
            list->_chunks[v.first] = v.second->dup();
        }

        return list;
    }

    bool NDArrayList::equals(NDArrayList& other) {
        if (_axis != other._axis)
            return false;

        if (_chunks.size() != other._chunks.size())
            return false;

        for (auto const& v : _chunks) {
            if (other._chunks.count(v.first) == 0)
                return false;

            auto arrThis = _chunks[v.first];
            auto arrThat = other._chunks[v.first];

            if (!arrThis->equalsTo(arrThat))
                return false;
        }

        return true;
    }
}