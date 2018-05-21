//
// Created by raver119 on 24.01.18.
//

#ifndef LIBND4J_INPUTLIST_H
#define LIBND4J_INPUTLIST_H

#include <op_boilerplate.h>
#include <pointercast.h>
#include <dll.h>
#include <vector>
#include <types/pair.h>

namespace nd4j {
namespace graph {
    class ND4J_EXPORT ArgumentsList {
    protected:
        std::vector<Pair> _arguments;
    public:
        explicit ArgumentsList() = default;
        ArgumentsList(std::initializer_list<Pair> arguments);
        ArgumentsList(std::initializer_list<int> arguments);

        ~ArgumentsList() = default;

        /**
         * This method returns number of argument pairs available
         *
         * @return
         */
        int size();

        /**
         * This method returns Pair at specified index
         *
         * @param index
         * @return
         */
        Pair &at(int index);
    };
}
}

#endif //LIBND4J_INPUTLIST_H
