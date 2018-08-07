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

#ifndef LIBND4J_SHAPELIST_H
#define LIBND4J_SHAPELIST_H

#include <vector>
#include <shape.h>

namespace nd4j {
    class ND4J_EXPORT ShapeList {
    protected:
        std::vector<Nd4jLong*> _shapes;

        bool _autoremovable = false;
        bool _workspace = false;
    public:
        ShapeList(Nd4jLong* shape = nullptr);
        ShapeList(std::initializer_list<Nd4jLong*> shapes);
        ShapeList(std::initializer_list<Nd4jLong*> shapes, bool isWorkspace);
        ShapeList(std::vector<Nd4jLong*>& shapes);
        //ShapeList(bool autoRemovable);

        ~ShapeList();

        std::vector<Nd4jLong*>* asVector();
        void destroy();
        int size();
        Nd4jLong* at(int idx);
        void push_back(Nd4jLong *shape);
        void push_back(std::vector<Nd4jLong>& shape);

        /**
         * PLEASE NOTE: This method should be called ONLY if shapes were generated at workspaces. Otherwise you'll get memory leak
         */
        void detach();
    };
}


#endif //LIBND4J_SHAPELIST_H
