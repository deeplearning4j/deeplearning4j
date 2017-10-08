//
// @author raver119@gmail.com
//

#include <indexing/NDIndex.h>

using namespace nd4j;

nd4j::NDIndexAll::NDIndexAll() : nd4j::NDIndex() {
    _indices.push_back(-1);
}

nd4j::NDIndexPoint::NDIndexPoint(int point) : nd4j::NDIndex() {
    this->_indices.push_back(point);
}


nd4j::NDIndexInterval::NDIndexInterval(int start, int end): nd4j::NDIndex() {
    for (int e = start; e < end; e++)
        this->_indices.push_back(e);
}

bool nd4j::NDIndex::isAll() {
    return _indices.size() == 1 && _indices.at(0) == -1;
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