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
        BUILD_TRACKER(OpType_TRANSFORM, TRANSFORM_OPS);
        BUILD_TRACKER(OpType_BROADCAST, BROADCAST_OPS);
        BUILD_TRACKER(OpType_PAIRWISE, PAIRWISE_TRANSFORM_OPS);
        BUILD_TRACKER(OpType_RANDOM, RANDOM_OPS);
        BUILD_TRACKER(OpType_ACCUMULATION, REDUCE_OPS);
        BUILD_TRACKER(OpType_ACCUMULATION3, REDUCE3_OPS);
        BUILD_TRACKER(OpType_INDEX_ACCUMULATION, INDEX_REDUCE_OPS);
        BUILD_TRACKER(OpType_SCALAR, SCALAR_OPS);
        BUILD_TRACKER(OpType_SUMMARYSTATS, SUMMARY_STATS_OPS);
//#endif
    };

    static nd4j::_loader loader;
}