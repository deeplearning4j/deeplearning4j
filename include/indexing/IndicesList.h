//
// @author raver119@gmail.com
//

#ifndef LIBND4J_INDICESLIST_H
#define LIBND4J_INDICESLIST_H

#include <initializer_list>
#include "NDIndex.h"

namespace nd4j {
    class IndicesList {
    protected:
        std::vector<NDIndex *> _indices;
    public:
        IndicesList(std::initializer_list<NDIndex *> list) {
            for (auto v: list)
                _indices.push_back(v);
        }

        int size();
        NDIndex* at(int idx);

        ~IndicesList() {
            for(auto v: _indices)
                delete v;
        }
    };
}

int nd4j::IndicesList::size() {
    return (int) _indices.size();
}

nd4j::NDIndex* nd4j::IndicesList::at(int idx) {
    return _indices.at(idx);
}

#endif //LIBND4J_INDICESLIST_H
