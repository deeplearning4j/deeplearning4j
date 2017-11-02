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
        int _stride = 1;
    public:
        NDIndex() = default;
        ~NDIndex() = default;

        bool isAll();

        std::vector<int>& getIndices();
        int stride();

        static NDIndex* all();
        static NDIndex* point(int pt);
        static NDIndex* interval(int start, int end, int stride = 1);
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
        NDIndexInterval(int start, int end, int stride = 1);

        ~NDIndexInterval() = default;
    };
}



#endif //LIBND4J_NDINDEX_H
