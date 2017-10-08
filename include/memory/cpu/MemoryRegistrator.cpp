//
// Created by raver119 on 07.10.2017.
//

#include <memory/MemoryRegistrator.h>

namespace nd4j {
    namespace memory {

        MemoryRegistrator::MemoryRegistrator() {
            _workspace = nullptr;
        };

        MemoryRegistrator* MemoryRegistrator::getInstance() {
            if (_INSTANCE == 0)
                _INSTANCE = new MemoryRegistrator();

            return _INSTANCE;
        }

        bool MemoryRegistrator::hasWorkspaceAttached() {
            return _workspace != nullptr;
        }

        Workspace* MemoryRegistrator::getWorkspace() {
            return _workspace;
        }

        void MemoryRegistrator::attachWorkspace(Workspace* workspace) {
            _workspace = workspace;
        }

        void MemoryRegistrator::forgetWorkspace() {
            _workspace = nullptr;
        }

        MemoryRegistrator* MemoryRegistrator::_INSTANCE = 0;

    }
}