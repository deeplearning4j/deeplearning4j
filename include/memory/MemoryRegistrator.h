//
// Created by raver119 on 12.09.17.
//

#ifndef LIBND4J_MEMORYREGISTRATOR_H
#define LIBND4J_MEMORYREGISTRATOR_H

#include "Workspace.h"
#include <map>
#include <mutex>

namespace nd4j {
    namespace memory {
        class MemoryRegistrator {
        protected:
            static MemoryRegistrator* _INSTANCE;
            Workspace* _workspace;
            std::map<Nd4jIndex, Nd4jIndex> _footprint;
            std::mutex _lock;

            MemoryRegistrator();
            ~MemoryRegistrator() = default;
        public:
            static MemoryRegistrator* getInstance();
            bool hasWorkspaceAttached();
            Workspace* getWorkspace();
            void attachWorkspace(Workspace* workspace);
            void forgetWorkspace();

            /**
             * This method allows you to set memory requirements for given graph
             */
            void setGraphMemoryFootprint(Nd4jIndex hash, Nd4jIndex bytes);

            /**
             * This method allows you to set memory requirements for given graph, ONLY if
             * new amount of bytes is greater then current one
             */
            void setGraphMemoryFootprintIfGreater(Nd4jIndex hash, Nd4jIndex bytes);

            /**
             * This method returns memory requirements for given graph
             */ 
            Nd4jIndex getGraphMemoryFootprint(Nd4jIndex hash);
        };
    }
}

#endif //LIBND4J_MEMORYREGISTRATOR_H
