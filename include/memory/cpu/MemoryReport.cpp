//
// Created by raver119 on 11.10.2017.
//

#include "../MemoryReport.h"

bool nd4j::memory::MemoryReport::operator<(const nd4j::memory::MemoryReport &other) const {
    return this->_rss < other._rss;
}

bool nd4j::memory::MemoryReport::operator>(const nd4j::memory::MemoryReport &other) const {
    return this->_rss > other._rss;
}

bool nd4j::memory::MemoryReport::operator==(const nd4j::memory::MemoryReport &other) const {
    return this->_rss == other._rss;
}

bool nd4j::memory::MemoryReport::operator!=(const nd4j::memory::MemoryReport &other) const {
    return this->_rss != other._rss;
}

bool nd4j::memory::MemoryReport::operator<=(const nd4j::memory::MemoryReport &other) const {
    return this->_rss <= other._rss;
}

bool nd4j::memory::MemoryReport::operator>=(const nd4j::memory::MemoryReport &other) const {
    return this->_rss >= other._rss;
}

Nd4jIndex nd4j::memory::MemoryReport::getVM() const {
    return _vm;
}

void nd4j::memory::MemoryReport::setVM(Nd4jIndex _vm) {
    MemoryReport::_vm = _vm;
}

Nd4jIndex nd4j::memory::MemoryReport::getRSS() const {
    return _rss;
}

void nd4j::memory::MemoryReport::setRSS(Nd4jIndex _rss) {
    MemoryReport::_rss = _rss;
}
