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

#include <array/ShapeList.h>

namespace sd {
// ShapeList::ShapeList(bool autoRemovable) {
//        _autoremovable = autoRemovable;
//    }

ShapeList::ShapeList(const sd::LongType* shape) {
  if (shape != nullptr) push_back(shape);
}

ShapeList::~ShapeList() {
  if (_autoremovable) destroy();
}

ShapeList::ShapeList(const std::vector<const sd::LongType*>& shapes, bool isWorkspace)
#if !defined(__NEC__)
    : ShapeList(shapes) {
#else
  {
  for (int i = 0; i < shapes.size(); i++) {
    push_back(shapes[i]);
  }
#endif
  _workspace = isWorkspace;
}

ShapeList::ShapeList(const std::vector<const sd::LongType*>& shapes) {
#if defined(__NEC__)
  for (int i = 0; i < shapes.size(); i++) {
    push_back(shapes[i]);
  }
#else
  _shapes = shapes;
#endif
}

void ShapeList::destroy() {

  if (_destroyed) return;
  if (!_workspace){
    for (int i = 0; i < size(); i++){
     // if (_shapes[i] != nullptr) delete[] _shapes[i];
    }
  }
  _destroyed = true;
}

int ShapeList::size() const {
#if defined(__NEC__)
  return size_x;
#else
  return (int)_shapes.size();
#endif
}

const sd::LongType* ShapeList::at(int idx) {

  if (size() <= idx || idx < 0) {
    std::string errorMessage;
    errorMessage += "Can't find requested variable by index: ";
    errorMessage += std::to_string(idx);
    THROW_EXCEPTION(errorMessage.c_str());
  }

  return _shapes[idx];
}

void ShapeList::push_back(const sd::LongType* shape) {
#if defined(__NEC__)
  if (size_x >= SD_MAX_INPUT_SIZE) {
    sd_printf("%s:%d Exceeded allowed limit of shapes.  ShapeList max size is (%d) \n", __FILE__, __LINE__,  SD_MAX_INPUT_SIZE);
    THROW_EXCEPTION("Exceeded allowed limit of shapes. ShapeList container for Nec has fixed size");
  }
  _shapes[size_x] = shape;
  ++size_x;
#else
  _shapes.push_back(shape);
#endif
}

void ShapeList::detach() {
  for (int e = 0; e < size(); e++) {
    _shapes[e] = shape::detachShape(_shapes[e]);
  }

  _autoremovable = true;
  _workspace = false;
}
}  // namespace sd
