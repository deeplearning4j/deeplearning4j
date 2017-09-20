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
        ArrayList() {
            //
        }

        ArrayList(const nd4j::graph::FlatResult* result) {
            for (int e = 0; e < result->variables()->size(); e++) {
                auto var = result->variables()->Get(e);

                std::vector<int> shapeInfo;
                for (int i = 0; i < var->shape()->size(); i++) {
                    shapeInfo.push_back(var->shape()->Get(i));
                }

                std::vector<int> shape;
                for (int i = 0; i < shapeInfo.at(0); i++) {
                    shape.push_back(shapeInfo.at(i+1));
                }

                auto array = new NDArray<T>((char) shapeInfo.at(shapeInfo.size() - 1), shape);

                for (int i = 0; i < var->values()->size(); i++) {
                    array->putScalar(i, var->values()->Get(i));
                }

                _content.push_back(array);
            }
        }



        // default destructor
        ~ArrayList() {
            for (auto v: _content)
                delete v;
        }


        int size() {
            return _content.size();
        }

        nd4j::NDArray<T> *at(unsigned long idx) {
            return _content.at(idx);
        }


        void push_back(nd4j::NDArray<T> *array) {
            _content.push_back(array);
        }
    };
}

#endif //LIBND4J_ARRAYLIST_H
