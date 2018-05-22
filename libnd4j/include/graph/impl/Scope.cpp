//
// Created by raver119 on 14.10.2017.
//

#include "Scope.h"

namespace nd4j {
    namespace graph {
        template <typename T>
        Scope<T>::Scope(int id, const char *name) {
            _id = id;

            if (name != nullptr)
                _name = name;
            else
                name = "";
        }

        template <typename T>
        Scope<T>::~Scope() {
            for (auto v: _nodes)
                delete v;
        }

        template <typename T>
        void Scope<T>::push_back(Node<T> *node) {
            _nodes.emplace_back(node);
        }

        template <typename T>
        std::vector<Node<T> *>* Scope<T>::nodes() {
            return &_nodes;
        }

        template <typename T>
        int Scope<T>::size() {
            return (int) _nodes.size();
        }

        template <typename T>
        int Scope<T>::id() {
            return _id;
        }

        template <typename T>
        std::string* Scope<T>::name() {
            return &_name;
        }

        template <typename T>
        void Scope<T>::forgetNodes() {
            _nodes.clear();
        }

        template <typename T>
        Scope<T>* Scope<T>::clone() {
            auto clone = new Scope<T>(_id, _name.c_str());

            for (auto v: _nodes)
                clone->_nodes.emplace_back(v->clone());

            return clone;
        }

        template <typename T>
        template <typename N>
        Scope<N>* Scope<T>::asT() {
            auto clone = new Scope<N>(_id, _name.c_str());

            for (auto v: _nodes)
                clone->push_back(v->template asT<N>());

            return clone;
        }

        template class Scope<float>;
        template class Scope<float16>;
        template class Scope<double>;


        template Scope<float>* Scope<float>::asT<float>();
        template Scope<float16>* Scope<float>::asT<float16>();
        template Scope<double>* Scope<float>::asT<double>();

        template Scope<float>* Scope<float16>::asT<float>();
        template Scope<float16>* Scope<float16>::asT<float16>();
        template Scope<double>* Scope<float16>::asT<double>();

        template Scope<float>* Scope<double>::asT<float>();
        template Scope<float16>* Scope<double>::asT<float16>();
        template Scope<double>* Scope<double>::asT<double>();
    }
}

