//
// Created by raver on 4/5/2018.
//

#ifndef LIBND4J_TRIPLE_H
#define LIBND4J_TRIPLE_H


#include <dll.h>

namespace nd4j {
    class ND4J_EXPORT Triple {
    protected:
        int _first = 0;
        int _second = 0;
        int _third = 0;

    public:
        Triple(int first = 0, int second = 0, int third = 0);
        ~Triple() = default;

        int first() const;
        int second() const;
        int third() const;
    };
}

#endif //LIBND4J_TRIPLE_H
