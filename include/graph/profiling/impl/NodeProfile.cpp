//
//  @author raver119@gmail.com
//

#include <helpers/logger.h>
#include <graph/profiling/NodeProfile.h>

namespace nd4j {
    namespace graph {
        NodeProfile::NodeProfile(int id, const char *name) {
            _id = id;

            if (name != nullptr)
                _name = name;
        };

        void NodeProfile::printOut() {
            nd4j_printf("Node: <%i:%s>\n", _id, _name.c_str());
            nd4j_printf("      Memory: ACT: %lld; TMP: %lld; OBJ: %lld; TTL: %lld;\n", _memoryActivations / _merges, _memoryTemporary / _merges, _memoryObjects / _merges, _memoryTotal / _merges);
            nd4j_printf("      Time: PREP: %lld ns; EXEC: %lld ns; TTL: %lld ns;\n", _preparationTime / _merges, _executionTime / _merges, _totalTime / _merges);
            nd4j_printf("      PREP: INPUT: %lld ns; SHAPE: %lld ns; ARRAY: %lld ns;\n", _inputTime / _merges, _shapeTime / _merges, _arrayTime / _merges);
        };

        Nd4jIndex NodeProfile::getActivationsSize() {
            return _memoryActivations;
        }

        void NodeProfile::setShapeFunctionTime(Nd4jIndex time) {
            _shapeTime = time;
        }

        void NodeProfile::setArrayTime(Nd4jIndex time) {
            _arrayTime = time;
        }

        void NodeProfile::setInputTime(Nd4jIndex time) {
            _inputTime = time;
        }

        Nd4jIndex NodeProfile::getTemporarySize() {
            return _memoryTemporary;
        }
            
        Nd4jIndex NodeProfile::getObjectsSize() {
            return _memoryObjects;
        }

        Nd4jIndex NodeProfile::getTotalSize() {
            return _memoryTotal;
        }

        void NodeProfile::setBuildTime(Nd4jIndex time) {
            _buildTime = time;
        }
        
        void NodeProfile::setPreparationTime(Nd4jIndex time) {
            _preparationTime = time;
        }
        
        void NodeProfile::setExecutionTime(Nd4jIndex time) {
            _executionTime = time;
        }

        void NodeProfile::setTotalTime(Nd4jIndex time) {
            _totalTime = time;
        }

        void NodeProfile::setActivationsSize(Nd4jIndex bytes) {
            _memoryActivations = bytes;
        }
            
        void NodeProfile::setTemporarySize(Nd4jIndex bytes) {
            _memoryTemporary = bytes;
        }
            
        void NodeProfile::setObjectsSize(Nd4jIndex bytes) {
            _memoryObjects = bytes;
        }

        void NodeProfile::setTotalSize(Nd4jIndex bytes) {
            _memoryTotal = bytes;
        }

        void NodeProfile::merge(NodeProfile *other) {
            _merges += other->_merges;
            _memoryObjects += other->_memoryObjects;
            _memoryActivations += other->_memoryActivations;
            _memoryTemporary += other->_memoryTemporary;
            _memoryTotal += other->_memoryTotal;

            _preparationTime += other->_preparationTime;
            _executionTime += other->_executionTime;
            _totalTime += other->_totalTime;
            _shapeTime += other->_shapeTime;
            _arrayTime += other->_arrayTime;
            _inputTime += other->_inputTime;
        }

        std::string& NodeProfile::name() {
            return _name;
        }

        void NodeProfile::assign(NodeProfile *other) {
            _merges = other->_merges;
            _memoryObjects = other->_memoryObjects;
            _memoryActivations = other->_memoryActivations;
            _memoryTemporary = other->_memoryTemporary;
            _memoryTotal = other->_memoryTotal;

            _preparationTime = other->_preparationTime;
            _executionTime = other->_executionTime;
            _totalTime = other->_totalTime;
            _shapeTime = other->_shapeTime;
            _arrayTime = other->_arrayTime;
            _inputTime = other->_inputTime;
        }
    }
}