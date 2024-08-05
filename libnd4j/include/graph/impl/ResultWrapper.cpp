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
// Created by raver119 on 11/06/18.
//
#include <graph/ResultWrapper.h>

#include <stdexcept>

namespace sd {
namespace graph {
ResultWrapper::ResultWrapper(LongType size, Pointer ptr) {
  if (size <= 0) THROW_EXCEPTION("FlatResult size should be > 0");

  _size = size;
  _pointer = ptr;
}

ResultWrapper::~ResultWrapper() {
  if (_pointer != nullptr && _size > 0) {
    auto ptr = reinterpret_cast<char *>(_pointer);
    delete[] ptr;
  }
}

LongType ResultWrapper::size() { return _size; }

Pointer ResultWrapper::pointer() { return _pointer; }
}  // namespace graph
}  // namespace sd
