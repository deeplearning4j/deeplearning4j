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
        NDIndex() {
            //
        }

        ~NDIndex() {
            //
        }

        bool isAll() {
            return _indices.size() == 1 && _indices.at(0) == -1;
        }
        std::vector<int>& getIndices();

        static NDIndex* all();
        static NDIndex* point(int pt);
        static NDIndex* interval(int start, int end);
    };

    class NDIndexAll : public NDIndex {
    public:
        NDIndexAll() : nd4j::NDIndex() {
            _indices.push_back(-1);
        }

        ~NDIndexAll() {
            //
        }
    };


    class NDIndexPoint : public NDIndex {
    public:
        explicit NDIndexPoint(int point): nd4j::NDIndex() {
            this->_indices.push_back(point);
        }

        ~NDIndexPoint() {
            //
        };
    };

    class NDIndexInterval : public NDIndex {
    public:
        explicit NDIndexInterval(int start, int end): nd4j::NDIndex() {
            for (int e = start; e < end; e++)
                this->_indices.push_back(e);
        }


        ~NDIndexInterval() {
            //
        }
    };
}

std::vector<int>& nd4j::NDIndex::getIndices() {
    return _indices;
}

nd4j::NDIndex* nd4j::NDIndex::all() {
    return new NDIndexAll();
}
nd4j::NDIndex* nd4j::NDIndex::point(int pt) {
    return new NDIndexPoint(pt);
}

nd4j::NDIndex* nd4j::NDIndex::interval(int start, int end) {
    return new NDIndexInterval(start, end);
}

#endif //LIBND4J_NDINDEX_H
