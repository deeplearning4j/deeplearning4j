//
// Created by raver119 on 24.01.18.
//

#include <graph/ArgumentsList.h>

namespace nd4j {
namespace graph {
    ArgumentsList::ArgumentsList(std::initializer_list<Pair> arguments) {
        _arguments = arguments;
    }

    ArgumentsList::ArgumentsList(std::initializer_list<int> arguments) {
        std::vector<int> args(arguments);
        for (int e = 0; e < args.size(); e++) {
            Pair pair(args[e]);
            _arguments.emplace_back(pair);
        }

    }

    int ArgumentsList::size() {
        return (int) _arguments.size();
    }

    Pair&  ArgumentsList::at(int index) {
        return _arguments.at(index);
    }
}
}
