//
//  @author raver119@gmail.com
//

#include <helpers/OpTracker.h>
#include <sstream>
#include <helpers/logger.h>

namespace nd4j {
    
    OpTracker* OpTracker::getInstance() {
        if (_INSTANCE == 0)
            _INSTANCE = new OpTracker();

        return _INSTANCE;
    }

    void OpTracker::storeOperation(nd4j::graph::OpType opType, const OpDescriptor& descriptor) {
        if (_map.count(opType) < 1) {
            std::vector<OpDescriptor> vec;
            _map[opType] = vec;
        }

        _operations++;

        auto vec = _map[opType];

        if (std::find(vec.begin(), vec.end(), descriptor) == vec.end())
            _map[opType].emplace_back(descriptor);
    }

    void OpTracker::storeOperation(nd4j::graph::OpType opType, const char* opName, const Nd4jLong opNum) {
        OpDescriptor descriptor(0, opName, false);
        descriptor.setOpNum((int) opNum);
        descriptor.setHash(-1);

        storeOperation(opType, descriptor);
    }


    template <typename T>
    std::string OpTracker::local_to_string(T value) {
        std::ostringstream os ;
        os << value ;
        return os.str() ;
    }


    int OpTracker::totalGroups() {
        return (int) _map.size();
    }

    int OpTracker::totalOperations() {
        return _operations;
    }

    const char* OpTracker::exportOperations() {
        if (_export.length() == 0) {
            for (auto &v: _map) {
                std::string block = local_to_string(v.first) + " ";

                for (auto &i: v.second) {
                    block += local_to_string(i.getHash()) + ":";
                    block += local_to_string(i.getOpNum()) + ":";
                    block += *i.getOpName() + "<<";
                }

                block += ">>";
                _export += block;
            }
        }

        return _export.c_str();
    }

    nd4j::OpTracker* nd4j::OpTracker::_INSTANCE = 0;
}