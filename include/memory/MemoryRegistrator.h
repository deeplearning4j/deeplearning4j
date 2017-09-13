//
// Created by raver119 on 12.09.17.
//

#ifndef LIBND4J_MEMORYREGISTRATOR_H
#define LIBND4J_MEMORYREGISTRATOR_H

#include "Workspace.h"

namespace nd4j {
    namespace memory {
        class MemoryRegistrator {
        protected:
            static MemoryRegistrator* _INSTANCE;
            Workspace* _workspace;

            MemoryRegistrator() {
                _workspace = nullptr;
            }

        public:
            ~MemoryRegistrator() {
                //
            }

            static MemoryRegistrator* getInstance() {
                if (_INSTANCE == 0) {
                    _INSTANCE = new MemoryRegistrator();
                }

                return _INSTANCE;
            }

            bool hasWorkspaceAttached() {
                return _workspace != nullptr;
            }

            Workspace* getWorkspace() {
                return _workspace;
            }

            void attachWorkspace(Workspace* workspace) {
                _workspace = workspace;
            }

            void forgetWorkspace() {
                _workspace = nullptr;
            }
        };
    }
}

nd4j::memory::MemoryRegistrator* nd4j::memory::MemoryRegistrator::_INSTANCE = 0;

#endif //LIBND4J_MEMORYREGISTRATOR_H
