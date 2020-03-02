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


#ifndef LIBND4J_OP_ENUMS_H
#define LIBND4J_OP_ENUMS_H

#include <loops/legacy_ops.h>
#include <system/type_boilerplate.h>
#include <system/enum_boilerplate.h>

namespace sd {
    namespace random {
        enum Ops {
            BUILD_ENUMERATION(RANDOM_OPS)
        };
    }

    namespace transform {
        enum FloatOps {
            BUILD_ENUMERATION(TRANSFORM_FLOAT_OPS)
        };

        enum SameOps {
            BUILD_ENUMERATION(TRANSFORM_SAME_OPS)
        };

        enum BoolOps {
            BUILD_ENUMERATION(TRANSFORM_BOOL_OPS)
        };

        enum AnyOps {
            BUILD_ENUMERATION(TRANSFORM_ANY_OPS)
        };

        enum StrictOps {
            BUILD_ENUMERATION(TRANSFORM_STRICT_OPS)
        };
    }

    namespace pairwise {
        enum Ops {
            BUILD_ENUMERATION(PAIRWISE_TRANSFORM_OPS)
        };

        enum BoolOps {
            BUILD_ENUMERATION(PAIRWISE_BOOL_OPS)
        };

        enum IntOps {
            BUILD_ENUMERATION(PAIRWISE_INT_OPS)
        };
    }

    namespace scalar {
        enum Ops {
            BUILD_ENUMERATION(SCALAR_OPS)
        };

        enum BoolOps {
            BUILD_ENUMERATION(SCALAR_BOOL_OPS)
        };

        enum IntOps {
            BUILD_ENUMERATION(SCALAR_INT_OPS)
        };
    }

    namespace reduce {
        enum FloatOps {
            BUILD_ENUMERATION(REDUCE_FLOAT_OPS)
        };

        enum SameOps {
            BUILD_ENUMERATION(REDUCE_SAME_OPS)
        };

        enum BoolOps {
            BUILD_ENUMERATION(REDUCE_BOOL_OPS)
        };

        enum LongOps {
            BUILD_ENUMERATION(REDUCE_LONG_OPS)
        };
    }

    namespace reduce3 {
        enum Ops {
            BUILD_ENUMERATION(REDUCE3_OPS)
        };
    }

    namespace indexreduce {
        enum Ops {
            BUILD_ENUMERATION(INDEX_REDUCE_OPS)
        };
    }

    namespace broadcast {
        enum Ops {
            BUILD_ENUMERATION(BROADCAST_OPS)
        };

        enum BoolOps {
            BUILD_ENUMERATION(BROADCAST_BOOL_OPS)
        };

        enum IntOps {
            BUILD_ENUMERATION(BROADCAST_INT_OPS)
        };
    }

    namespace variance {
        enum Ops {
            BUILD_ENUMERATION(SUMMARY_STATS_OPS)
        };
    }

    namespace logic {
        enum Ops {
            While = 0,
            Scope = 10,
            Conditional = 20,
            Switch = 30,
            Return = 40,
            Expose = 50,
            Merge = 60,
            LoopCond = 70,
            NextIteration = 80,
            Exit = 90,
            Enter = 100,
        };
    }
}

#endif