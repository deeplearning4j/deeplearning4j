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
//  @author raver119@gmail.com
//

#include <type_boilerplate.h>

namespace nd4j {
    enum TransformOps {
       Add,
       Sum,
    };

    enum PairwiseOps {
       Add,
       Sum,
    };

    enum ScalarOps {
        Add,
        Multiply,
        Subtract,
    };

    enum ReduceOps {
        Sum,
        Mean,
    };

    enum Reduce3Ops {
        CosineDistance,
    };

    enum IndexReduceOps {
        IndexFirst,
        IndexLast,
    };

    enum BroadcastOps {
        Add,
        Subtract,
    };

    enum VarianceOps {
        Variance,
        StandardVariance,
    };
}