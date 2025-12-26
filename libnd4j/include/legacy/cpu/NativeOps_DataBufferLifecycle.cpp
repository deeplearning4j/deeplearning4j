#include <legacy/NativeOps.h>
#include <array/DataBufferLifecycleTracker.h>
#include <array/DataBuffer.h>
#include <vector>

SD_LIB_EXPORT void recordJavaDataBufferAllocation(OpaqueDataBuffer *buffer, long size, int dataType, bool isWorkspace) {
#if defined(SD_GCC_FUNCTRACE)
    if (buffer != nullptr) {
        sd::array::DataBufferLifecycleTracker::getInstance().recordAllocation(
            buffer->primary(), size, (sd::DataType)dataType, sd::array::BufferType::PRIMARY, buffer, isWorkspace, sd::array::DataBufferSegment::JAVA);
    }
#endif
}

SD_LIB_EXPORT void recordJavaDataBufferDeallocation(OpaqueDataBuffer *buffer) {
#if defined(SD_GCC_FUNCTRACE)
    if (buffer != nullptr) {
        sd::array::DataBufferLifecycleTracker::getInstance().recordDeallocation(buffer->primary(), sd::array::BufferType::PRIMARY);
    }
#endif
}
