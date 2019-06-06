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
// Created by raver119 on 11.10.2017.
//

#ifndef LIBND4J_MEMORYREPORT_H
#define LIBND4J_MEMORYREPORT_H

#include <pointercast.h>

namespace nd4j {
    namespace memory {
        class MemoryReport {
        private:
            Nd4jLong _vm = 0;
            Nd4jLong _rss = 0;

        public:
            MemoryReport() = default;
            ~MemoryReport() = default;

            bool operator < (const MemoryReport& other) const;
            bool operator <= (const MemoryReport& other) const;
            bool operator > (const MemoryReport& other) const;
            bool operator >= (const MemoryReport& other) const;
            bool operator == (const MemoryReport& other) const;
            bool operator != (const MemoryReport& other) const;

            Nd4jLong getVM() const;
            void setVM(Nd4jLong vm);

            Nd4jLong getRSS() const;
            void setRSS(Nd4jLong rss);
        };
    }
}



#endif //LIBND4J_MEMORYREPORT_H
