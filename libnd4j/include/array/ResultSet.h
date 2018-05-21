//
// This class is suited for execution results representation. 
// 
// PLESE NOTE: It will delete all stored NDArrays upon destructor call
//
// Created by raver119 on 07.09.17.
//

#ifndef LIBND4J_RESULTSET_H
#define LIBND4J_RESULTSET_H

#include <NDArray.h>
#include <vector>
#include <graph/generated/result_generated.h>

namespace  nd4j {
    template<typename T>
    class ResultSet {
    private:
        std::vector<nd4j::NDArray<T> *> _content;
        Nd4jStatus _status = ND4J_STATUS_OK;
        bool _removable = true;

    public:
        // default constructor
        ResultSet(const nd4j::graph::FlatResult* result = nullptr);
        ~ResultSet();

        int size();
        nd4j::NDArray<T> *at(unsigned long idx);
        void push_back(nd4j::NDArray<T> *array);

        Nd4jStatus status();
        void setStatus(Nd4jStatus status);
        void purge();
        void setNonRemovable();
    };
}

#endif //LIBND4J_RESULTSET_H
