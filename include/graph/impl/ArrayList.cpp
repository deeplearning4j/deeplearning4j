//
// @author raver119@gmail.com
//

#include <graph/ArrayList.h>

namespace nd4j {
    template <typename T>
    ArrayList<T>::ArrayList(const nd4j::graph::FlatResult* result) {
        if (result != nullptr) {
            for (int e = 0; e < result->variables()->size(); e++) {
                auto var = result->variables()->Get(e);

                std::vector<int> shapeInfo;
                for (int i = 0; i < var->shape()->size(); i++) {
                    shapeInfo.push_back(var->shape()->Get(i));
                }

                std::vector<int> shape;
                for (int i = 0; i < shapeInfo.at(0); i++) {
                    shape.push_back(shapeInfo.at(i + 1));
                }

                auto array = new NDArray<T>((char) shapeInfo.at(shapeInfo.size() - 1), shape);

                for (int i = 0; i < var->values()->size(); i++) {
                    array->putScalar(i, var->values()->Get(i));
                }

                _content.push_back(array);
            }
        }
    }

    template <typename T>
    ArrayList<T>::~ArrayList() {
        for (auto v: _content)
            delete v;
    }

    template <typename T>
    int ArrayList<T>::size() {
        return _content.size();
    }

    template <typename T>
    nd4j::NDArray<T>* ArrayList<T>::at(unsigned long idx) {
        return _content.at(idx);
    }

    template <typename T>
    void ArrayList<T>::push_back(nd4j::NDArray<T> *array) {
        _content.push_back(array);
    }

    template class ArrayList<float>;
    template class ArrayList<float16>;
    template class ArrayList<double>;
}

