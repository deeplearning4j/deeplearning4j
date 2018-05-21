//
// Created by raver119 on 15/11/17.
//

#ifndef LIBND4J_VARIABLESSET_H
#define LIBND4J_VARIABLESSET_H

#include <iterator>
#include <vector>
#include <pointercast.h>
#include <dll.h>
#include <graph/Variable.h>


namespace nd4j {
    namespace graph {
        template <typename T>
        class VariablesSet {
        protected:
            std::vector<nd4j::graph::Variable<T>*> _holder;
            Nd4jStatus _status;
        public:
            VariablesSet(Nd4jStatus status = ND4J_STATUS_OK);
            ~VariablesSet();

            Nd4jStatus status();

            int size();

            void push_back(Variable<T>* variable);

            Variable<T>* at(int index);

        };
    }
}



#endif //LIBND4J_VARIABLESSET_H
