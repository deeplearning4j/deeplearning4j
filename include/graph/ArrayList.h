//
// Created by raver119 on 07.09.17.
//

#ifndef LIBND4J_ARRAYLIST_H
#define LIBND4J_ARRAYLIST_H

#include <NDArray.h>

template <typename T>
class ArrayList {
private:
    std::vector<nd4j::NDArray<T>*> _content;

public:
    // default constructor
    ArrayList() {
        //
    }

    // default destructor
    ~ArrayList() {
        for(auto v: _content)
            delete v;
    }


    int size() {
        return _content.size();
    }

    nd4j::NDArray<T>* at(unsigned long idx) {
        return _content.at(idx);
    }


    void push_back(nd4j::NDArray<T>* array) {
        _content.push_back(array);
    }
};

#endif //LIBND4J_ARRAYLIST_H
