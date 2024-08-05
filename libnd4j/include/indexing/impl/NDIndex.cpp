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
#include <indexing/NDIndex.h>

namespace sd {

bool NDIndex::isInterval() { return false; }

LongType NDIndex::stride() { return _stride; }

NDIndexAll::NDIndexAll() : NDIndex() { _indices.push_back(-1); }

NDIndexPoint::NDIndexPoint(LongType point) : NDIndex() { this->_indices.push_back(point); }

bool NDIndexAll::isInterval() { return false; }

bool NDIndexPoint::isInterval() { return false; }

bool NDIndexInterval::isInterval() { return true; }

NDIndexInterval::NDIndexInterval(LongType start, LongType end, LongType stride) : NDIndex() {
  this->_stride = stride;
  for (int e = start; e < end; e += stride) this->_indices.push_back(e);
}

bool NDIndex::isAll() { return _indices.size() == 1 && _indices.at(0) == -1; }

bool NDIndex::isPoint() { return _indices.size() == 1 && _indices.at(0) >= 0; }

std::vector<LongType> &NDIndex::getIndices() { return _indices; }

NDIndex *NDIndex::all() { return new NDIndexAll(); }

NDIndex *NDIndex::point(LongType pt) { return new NDIndexPoint(pt); }

NDIndex *NDIndex::interval(LongType start, LongType end, LongType stride) {
  return new NDIndexInterval(start, end, stride);
}
}  // namespace sd
