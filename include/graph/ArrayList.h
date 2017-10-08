//
// Created by raver119 on 07.09.17.
//

#ifndef LIBND4J_ARRAYLIST_H
#define LIBND4J_ARRAYLIST_H

#include <NDArray.h>
#include <vector>
#include <graph/generated/result_generated.h>

namespace  nd4j {
    template<typename T>
    class ArrayList {
    private:
        std::vector<nd4j::NDArray<T> *> _content;

    public:
        // default constructor
        ArrayList(const nd4j::graph::FlatResult* result = nullptr);
        ~ArrayList();

        int size();
        nd4j::NDArray<T> *at(unsigned long idx);
        void push_back(nd4j::NDArray<T> *array);
    };
}

#endif //LIBND4J_ARRAYLIST_H
