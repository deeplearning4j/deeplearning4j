//
// @author raver119@gmail.com
//

#ifndef LIBND4J_NODE_PROFILE_H
#define LIBND4J_NODE_PROFILE_H

#include <pointercast.h>
#include <dll.h>
#include <string>

namespace nd4j {
    namespace graph {
        class ND4J_EXPORT NodeProfile {
        private:
            int _id;
            std::string _name;

            Nd4jIndex _merges = 1L;

            // time spent during deserialization
            Nd4jIndex _buildTime = 0L;
            
            // time spent before op execution
            Nd4jIndex _preparationTime = 0L;

            // time spent for op execution
            Nd4jIndex _executionTime = 0L;

            // total time spent during node execution
            Nd4jIndex _totalTime = 0L;

            // amount of memory used for outputs
            Nd4jIndex _memoryActivations = 0L;

            // amount of memory used internally for temporary arrays
            Nd4jIndex _memoryTemporary = 0L;

            // amount of memory used internally for objects
            Nd4jIndex _memoryObjects = 0L;
        public:
            NodeProfile() = default;
            ~NodeProfile() = default;

            explicit NodeProfile(int id, const char *name);

            void setBuildTime(Nd4jIndex time);
            void setPreparationTime(Nd4jIndex time);
            void setExecutionTime(Nd4jIndex time);
            void setTotalTime(Nd4jIndex time);

            void setActivationsSize(Nd4jIndex bytes);
            void setTemporarySize(Nd4jIndex bytes);
            void setObjectsSize(Nd4jIndex bytes);

            Nd4jIndex getActivationsSize();
            Nd4jIndex getTemporarySize();
            Nd4jIndex getObjectsSize();

            std::string& name();

            void merge(NodeProfile *other);
            void assign(NodeProfile *other);

            void printOut();
        };
    }
}

#endif