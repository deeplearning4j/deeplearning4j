//
// @author raver119@gmail.com
//

#ifndef LIBND4J_NDINDEX_H
#define LIBND4J_NDINDEX_H

#include <vector>
#include <dll.h>

namespace nd4j {
    class ND4J_EXPORT NDIndex {
    protected:
        std::vector<int> _indices;
    public:
        NDIndex() = default;
        ~NDIndex() = default;

        bool isAll();

        std::vector<int>& getIndices();

        static NDIndex* all();
        static NDIndex* point(int pt);
        static NDIndex* interval(int start, int end);
    };

    class ND4J_EXPORT NDIndexAll : public NDIndex {
    public:
        NDIndexAll();

        ~NDIndexAll() = default;
    };


    class ND4J_EXPORT NDIndexPoint : public NDIndex {
    public:
        NDIndexPoint(int point);

        ~NDIndexPoint() = default;
    };

    class ND4J_EXPORT NDIndexInterval : public NDIndex {
    public:
        NDIndexInterval(int start, int end);

        ~NDIndexInterval() = default;
    };
}



#endif //LIBND4J_NDINDEX_H
