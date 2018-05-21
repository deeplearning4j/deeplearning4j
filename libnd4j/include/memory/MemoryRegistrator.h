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
            std::map<Nd4jLong, Nd4jLong> _footprint;
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
            void setGraphMemoryFootprint(Nd4jLong hash, Nd4jLong bytes);

            /**
             * This method allows you to set memory requirements for given graph, ONLY if
             * new amount of bytes is greater then current one
             */
            void setGraphMemoryFootprintIfGreater(Nd4jLong hash, Nd4jLong bytes);

            /**
             * This method returns memory requirements for given graph
             */ 
            Nd4jLong getGraphMemoryFootprint(Nd4jLong hash);
        };
    }
}

#endif //LIBND4J_MEMORYREGISTRATOR_H
