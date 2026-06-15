#include <legacy/NativeOps.h>
#include <graph/OpContextLifecycleTracker.h>
#include <graph/Context.h>


void recordJavaOpContextAllocation(OpaqueContext *context, int nodeId, long fastpathInSize, long fastpathOutSize, long intermediateResultsSize, long handlesSize, bool hasWorkspace, bool isFastPath) {
#if defined(SD_GCC_FUNCTRACE)
    if (context != nullptr) {
        sd::graph::OpContextLifecycleTracker::getInstance().recordAllocation(
            context, nodeId, fastpathInSize, fastpathOutSize, intermediateResultsSize, handlesSize, hasWorkspace, isFastPath, sd::graph::OpContextSegment::JAVA);
    }
#endif
}

void recordJavaOpContextDeallocation(OpaqueContext *context) {
#if defined(SD_GCC_FUNCTRACE)
    if (context != nullptr) {
        sd::graph::OpContextLifecycleTracker::getInstance().recordDeallocation(context);
    }
#endif
}



