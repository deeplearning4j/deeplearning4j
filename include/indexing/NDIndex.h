//
// @author raver119@gmail.com
//

#ifndef LIBND4J_NDINDEX_H
#define LIBND4J_NDINDEX_H

#include <vector>

namespace nd4j {
    class NDIndex {
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

    class NDIndexAll : public NDIndex {
    public:
        NDIndexAll();

        ~NDIndexAll() = default;
    };


    class NDIndexPoint : public NDIndex {
    public:
        NDIndexPoint(int point);

        ~NDIndexPoint() = default;
    };

    class NDIndexInterval : public NDIndex {
    public:
        NDIndexInterval(int start, int end);

        ~NDIndexInterval() = default;
    };
}



#endif //LIBND4J_NDINDEX_H
