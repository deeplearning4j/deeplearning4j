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

nd4j::NDIndex* nd4j::IndicesList::at(int idx) {
    return _indices.at(idx);
}
