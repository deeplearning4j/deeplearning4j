#include <legacy/NativeOps.h>
#include <graph/OpContextLifecycleTracker.h>
#include <graph/Context.h>


SD_LIB_EXPORT void recordJavaOpContextAllocation(OpaqueContext *context, int nodeId, long fastpathInSize, long fastpathOutSize, long intermediateResultsSize, long handlesSize, bool hasWorkspace, bool isFastPath) {
#if defined(SD_GCC_FUNCTRACE)
    if (context != nullptr) {
        sd::graph::OpContextLifecycleTracker::getInstance().recordAllocation(
            context, nodeId, fastpathInSize, fastpathOutSize, intermediateResultsSize, handlesSize, hasWorkspace, isFastPath, sd::graph::OpContextSegment::JAVA);
    }
#endif
}

SD_LIB_EXPORT void recordJavaOpContextDeallocation(OpaqueContext *context) {
#if defined(SD_GCC_FUNCTRACE)
    if (context != nullptr) {
        sd::graph::OpContextLifecycleTracker::getInstance().recordDeallocation(context);
    }
#endif
}
