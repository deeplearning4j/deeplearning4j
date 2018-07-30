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

#ifndef LIBND4J_NDINDEX_H
#define LIBND4J_NDINDEX_H

#include <pointercast.h>
#include <vector>
#include <dll.h>

namespace nd4j {
    class ND4J_EXPORT NDIndex {
    protected:
        std::vector<Nd4jLong> _indices;
        Nd4jLong _stride = 1;
    public:
        NDIndex() = default;
        ~NDIndex() = default;

        bool isAll();
        bool isPoint();

        std::vector<Nd4jLong>& getIndices();
        Nd4jLong stride();

        static NDIndex* all();
        static NDIndex* point(Nd4jLong pt);
        static NDIndex* interval(Nd4jLong start, Nd4jLong end, Nd4jLong stride = 1);
    };

    class ND4J_EXPORT NDIndexAll : public NDIndex {
    public:
        NDIndexAll();

        ~NDIndexAll() = default;
    };


    class ND4J_EXPORT NDIndexPoint : public NDIndex {
    public:
        NDIndexPoint(Nd4jLong point);

        ~NDIndexPoint() = default;
    };

    class ND4J_EXPORT NDIndexInterval : public NDIndex {
    public:
        NDIndexInterval(Nd4jLong start, Nd4jLong end, Nd4jLong stride = 1);

        ~NDIndexInterval() = default;
    };
}



#endif //LIBND4J_NDINDEX_H
