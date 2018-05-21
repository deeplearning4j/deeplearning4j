//
// @author raver119@gmail.com
//

#include <indexing/IndicesList.h>

using namespace nd4j;

nd4j::IndicesList::IndicesList(std::initializer_list<NDIndex *> list) {
	for (auto v: list)
	_indices.emplace_back(v);
}

nd4j::IndicesList::~IndicesList() {
    for(auto v: _indices)
        delete v;
}

int nd4j::IndicesList::size() {
    return (int) _indices.size();
}

bool nd4j::IndicesList::isScalar() {
    if (_indices.size() == 1) {
        return _indices.at(0)->isPoint();
    }

    return false;
}

nd4j::NDIndex* nd4j::IndicesList::at(int idx) {
    return _indices.at(idx);
}

void nd4j::IndicesList::push_back(NDIndex* idx) {
    _indices.emplace_back(idx);
}