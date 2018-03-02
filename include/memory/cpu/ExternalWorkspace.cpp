//
//  @author raver119@gmail.com
//

#include <memory/ExternalWorkspace.h>

namespace nd4j {
    namespace memory {
        ExternalWorkspace::ExternalWorkspace(Nd4jPointer ptrH, Nd4jIndex sizeH, Nd4jPointer ptrD, Nd4jIndex sizeD) {
            _ptrH = ptrH;
            _sizeH = sizeH;

            _ptrD = ptrD;
            _sizeD = sizeD;
        };

        void* ExternalWorkspace::pointerHost() {
            return _ptrH;
        }

        void* ExternalWorkspace::pointerDevice() {
            return _ptrD;
        }

        Nd4jIndex ExternalWorkspace::sizeHost() {
            return _sizeH;
        }
        
        Nd4jIndex ExternalWorkspace::sizeDevice() {
            return _sizeD;
        }
    }
}