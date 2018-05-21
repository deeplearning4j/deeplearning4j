//
// Created by raver119 on 11.10.2017.
//

#ifndef LIBND4J_MEMORYUTILS_H
#define LIBND4J_MEMORYUTILS_H

#include "MemoryReport.h"

namespace nd4j {
    namespace memory {
        class MemoryUtils {
        public:
            static bool retrieveMemoryStatistics(MemoryReport& report);
        };
    }
}



#endif //LIBND4J_MEMORYUTILS_H
