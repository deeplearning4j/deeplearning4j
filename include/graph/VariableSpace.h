//
// @author raver119@gmail.com
//

#ifndef LIBND4J_VARIABLESPACE_H
#define LIBND4J_VARIABLESPACE_H

#include <string>
#include <map>
#include <NDArray.h>
#include <graph/Variable.h>

namespace nd4j {
    namespace graph {

        template <typename T>
        class VariableSpace {
        protected:
            std::map<int32_t, nd4j::graph::Variable<T> *> *_variables;

        public:
            VariableSpace();
            ~VariableSpace();
        };
    }
}


nd4j::graph::VariableSpace::~VariableSpace() {
    _variables = new std::map<int32_t, nd4j::graph::Variable<T> *>();
}

nd4j::graph::VariableSpace::VariableSpace() {
    // TODO: loop through variables and release them

    delete _variables;
}

#endif //LIBND4J_VARIABLESPACE_H
