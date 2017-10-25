//
// @author raver119@gmail.com
//

#ifndef LIBND4J_VARIABLESPACE_H
#define LIBND4J_VARIABLESPACE_H

#include <helpers/logger.h>
#include <string>
#include <vector>
#include <list>
#include <map>
#include <mutex>
//#include <NDArray.h>
#include <graph/Variable.h>
#include <memory/Workspace.h>
#include <graph/Stash.h>

namespace nd4j {
    namespace graph {

        template <typename T>
        class VariableSpace {
        protected:

            nd4j::memory::Workspace _workspace;

            // stash is NOT cloned
            nd4j::graph::Stash<T> _stash;

            std::map<std::pair<int, int>, nd4j::graph::Variable<T> *> _paired;
            std::map<std::string, nd4j::graph::Variable<T> *> _symbolic;
            std::map<int, nd4j::graph::Variable<T> *> _variables;
            std::vector<nd4j::graph::Variable<T> *> _external;
            std::vector<nd4j::graph::Variable<T> *> _internal;

            std::vector<nd4j::graph::Variable<T> *> _placeholders;

            void silentPutVariable(std::pair<int,int>& pair, Variable<T> *variable);

            int _auto_counter = -1;

            std::mutex _varmap;

            std::map<int, nd4j::graph::Variable<T> *> _temporary;

            std::vector<nd4j::graph::Variable<T> *> *_handles;
        public:
            VariableSpace();
            ~VariableSpace();

            int numberOfPlaceholders();
            std::vector<nd4j::graph::Variable<T>*>* getPlaceholders();

            bool hasVariable(int id);
            bool hasVariable(int id, int idx);
            bool hasVariable(std::pair<int,int>& pair);
            bool hasVariable(std::string *symbol);

            nd4j::graph::Variable<T> *getVariable(int id);
            nd4j::graph::Variable<T> *getVariable(int id, int idx);
            nd4j::graph::Variable<T> *getVariable(std::pair<int,int>& pair);
            nd4j::graph::Variable<T> *getVariable(std::string *symbol);

            void putVariable(std::pair<int,int>& pair, NDArray<T> *array);
            void putVariable(std::pair<int,int>& pair, Variable<T> *variable);
            void putVariable(int id, Variable<T> *variable);
            void putVariable(int id, NDArray<T> *array);


            void putOutputVariable(Variable<T> *variable);

            // memory-related statistics
            Nd4jIndex externalMemory();
            Nd4jIndex internalMemory();
            Nd4jIndex totalMemory();

            int externalEntries();
            int internalEntries();
            int totalEntries();

            nd4j::graph::VariableSpace<T>* clone();

            nd4j::graph::Stash<T>* getStash();

            std::vector<nd4j::graph::Variable<T> *> * getExternalVariables();


        };
    }
}


#endif //LIBND4J_VARIABLESPACE_H
