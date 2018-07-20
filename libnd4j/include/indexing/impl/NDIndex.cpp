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

#include <indexing/NDIndex.h>

namespace nd4j {

    Nd4jLong NDIndex::stride() {
        return _stride;
    }

    nd4j::NDIndexAll::NDIndexAll() : nd4j::NDIndex() {
        _indices.push_back(-1);
    }

    nd4j::NDIndexPoint::NDIndexPoint(Nd4jLong point) : nd4j::NDIndex() {
        this->_indices.push_back(point);
    }


    nd4j::NDIndexInterval::NDIndexInterval(Nd4jLong start, Nd4jLong end, Nd4jLong stride) : nd4j::NDIndex() {
        this->_stride = stride;
        for (int e = start; e < end; e+= stride)
            this->_indices.push_back(e);
    }

    bool nd4j::NDIndex::isAll() {
        return _indices.size() == 1 && _indices.at(0) == -1;
    }

    bool nd4j::NDIndex::isPoint() {
        return _indices.size() == 1 && _indices.at(0) >= 0;
    }

    std::vector<Nd4jLong> &nd4j::NDIndex::getIndices() {
        return _indices;
    }


    nd4j::NDIndex *nd4j::NDIndex::all() {
        return new NDIndexAll();
    }

    nd4j::NDIndex *nd4j::NDIndex::point(Nd4jLong pt) {
        return new NDIndexPoint(pt);
    }

    nd4j::NDIndex *nd4j::NDIndex::interval(Nd4jLong start, Nd4jLong end, Nd4jLong stride) {
        return new NDIndexInterval(start, end, stride);
    }
}