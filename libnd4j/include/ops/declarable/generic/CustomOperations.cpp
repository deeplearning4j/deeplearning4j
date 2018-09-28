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
// This is special snowflake. This file builds bindings for ops availability tests
//
// @author raver119@gmail.com
//

#include <loops/legacy_ops.h>
#include <helpers/OpTracker.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {

    _loader::_loader() {
        //
        OpTracker::getInstance();

//#ifndef __CLION_IDE__
        BUILD_TRACKER(OpType_TRANSFORM, TRANSFORM_FLOAT_OPS);
        BUILD_TRACKER(OpType_TRANSFORM, TRANSFORM_SAME_OPS);
        BUILD_TRACKER(OpType_TRANSFORM, TRANSFORM_BOOL_OPS);
        BUILD_TRACKER(OpType_BROADCAST, BROADCAST_OPS);
        BUILD_TRACKER(OpType_PAIRWISE, PAIRWISE_TRANSFORM_OPS);
        BUILD_TRACKER(OpType_RANDOM, RANDOM_OPS);
        BUILD_TRACKER(OpType_ACCUMULATION, REDUCE_FLOAT_OPS);
        BUILD_TRACKER(OpType_ACCUMULATION, REDUCE_SAME_OPS);
        BUILD_TRACKER(OpType_ACCUMULATION, REDUCE_BOOL_OPS);
        BUILD_TRACKER(OpType_ACCUMULATION3, REDUCE3_OPS);
        BUILD_TRACKER(OpType_INDEX_ACCUMULATION, INDEX_REDUCE_OPS);
        BUILD_TRACKER(OpType_SCALAR, SCALAR_OPS);
        BUILD_TRACKER(OpType_SUMMARYSTATS, SUMMARY_STATS_OPS);
//#endif
    };

    static nd4j::_loader loader;
}