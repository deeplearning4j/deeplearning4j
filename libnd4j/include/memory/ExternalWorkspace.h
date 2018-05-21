//
//  @author raver119@gmail.com
//

#ifndef LIBND4J_EXTERNALWORKSPACE_H
#define LIBND4J_EXTERNALWORKSPACE_H

#include <pointercast.h>
#include <dll.h>

namespace nd4j {
    namespace memory {
        class ND4J_EXPORT ExternalWorkspace {
        private:
            void *_ptrH = nullptr;
            void *_ptrD = nullptr;

            Nd4jLong _sizeH = 0L;
            Nd4jLong _sizeD = 0L;
        public:
            ExternalWorkspace() = default;
            ~ExternalWorkspace() = default;

            ExternalWorkspace(Nd4jPointer ptrH, Nd4jLong sizeH, Nd4jPointer ptrD, Nd4jLong sizeD);
            
            void *pointerHost();
            void *pointerDevice();

            Nd4jLong sizeHost();
            Nd4jLong sizeDevice();
        };
    }
}

#endif