#include <legacy/NativeOps.h>
#include <array/NDArrayLifecycleTracker.h>
#include <array/NDArray.h>
#include <vector>

SD_LIB_EXPORT void recordJavaNDArrayAllocation(OpaqueNDArray array, long size, int dataType, bool isView) {
#if defined(SD_GCC_FUNCTRACE)
    auto ptr = reinterpret_cast<sd::NDArray*>(array);
    std::vector<sd::LongType> shapeVector;
    if(ptr != nullptr && ptr->shapeInfo() != nullptr) {
        for(int i = 0; i < ptr->rankOf(); i++) {
            shapeVector.push_back(ptr->shapeOf()[i]);
        }
    }
    sd::array::NDArrayLifecycleTracker::getInstance().recordAllocation(
        array, size, (sd::DataType)dataType, shapeVector, isView, sd::array::NDArraySegment::JAVA);
#endif
}

SD_LIB_EXPORT void recordJavaNDArrayDeallocation(OpaqueNDArray array) {
#if defined(SD_GCC_FUNCTRACE)
    sd::array::NDArrayLifecycleTracker::getInstance().recordDeallocation(array);
#endif
}
