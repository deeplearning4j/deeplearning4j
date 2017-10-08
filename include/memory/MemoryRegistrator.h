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

            MemoryRegistrator();
            ~MemoryRegistrator() = default;


        public:
            static MemoryRegistrator* getInstance();
            bool hasWorkspaceAttached();
            Workspace* getWorkspace();
            void attachWorkspace(Workspace* workspace);
            void forgetWorkspace();
        };
    }
}

#endif //LIBND4J_MEMORYREGISTRATOR_H
