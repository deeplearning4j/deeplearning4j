//
// @author raver119@gmail.com
//

#ifndef LIBND4J_INDICESLIST_H
#define LIBND4J_INDICESLIST_H

#include <initializer_list>
#include "NDIndex.h"

namespace nd4j {
    class ND4J_EXPORT IndicesList {
    protected:
        std::vector<NDIndex *> _indices;
    public:
        explicit IndicesList(std::initializer_list<NDIndex *> list = {});

        int size();
        NDIndex* at(int idx);
        void push_back(NDIndex* idx);
        bool isScalar();

        ~IndicesList();
    };
}
#endif //LIBND4J_INDICESLIST_H
