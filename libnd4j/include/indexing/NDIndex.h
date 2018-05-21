//
// @author raver119@gmail.com
//

#ifndef LIBND4J_NDINDEX_H
#define LIBND4J_NDINDEX_H

#include <pointercast.h>
#include <vector>
#include <dll.h>

namespace nd4j {
    class ND4J_EXPORT NDIndex {
    protected:
        std::vector<Nd4jLong> _indices;
        Nd4jLong _stride = 1;
    public:
        NDIndex() = default;
        ~NDIndex() = default;

        bool isAll();
        bool isPoint();

        std::vector<Nd4jLong>& getIndices();
        Nd4jLong stride();

        static NDIndex* all();
        static NDIndex* point(Nd4jLong pt);
        static NDIndex* interval(Nd4jLong start, Nd4jLong end, Nd4jLong stride = 1);
    };

    class ND4J_EXPORT NDIndexAll : public NDIndex {
    public:
        NDIndexAll();

        ~NDIndexAll() = default;
    };


    class ND4J_EXPORT NDIndexPoint : public NDIndex {
    public:
        NDIndexPoint(Nd4jLong point);

        ~NDIndexPoint() = default;
    };

    class ND4J_EXPORT NDIndexInterval : public NDIndex {
    public:
        NDIndexInterval(Nd4jLong start, Nd4jLong end, Nd4jLong stride = 1);

        ~NDIndexInterval() = default;
    };
}



#endif //LIBND4J_NDINDEX_H
