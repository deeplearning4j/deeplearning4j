//
// @author raver119@gmail.com
//

#ifndef LIBND4J_STASH_H
#define LIBND4J_STASH_H

//#include <graph/Block.h>
#include <NDArray.h>
#include <map>
#include <string>
#include <atomic>
#include <pointercast.h>

namespace nd4j {
    namespace graph {
        class ND4J_EXPORT KeyPair {
            int _node;
            std::string _name;
        public:
            KeyPair(int node = 0, const char * name = nullptr);

            bool operator<(const KeyPair& other) const;
        };

        template <typename T>
        class Stash {
        protected:
            std::map<nd4j::graph::KeyPair, nd4j::NDArray<T>*> _stash;
            std::vector<nd4j::NDArray<T>*> _handles;

        public:
            Stash();
            ~Stash();

            //void storeArray(nd4j::graph::Block<T>& block, const char *name, nd4j::NDArray<T> *array);
            void storeArray(int nodeId, const char *name, nd4j::NDArray<T> *array);

            //bool checkStash(nd4j::graph::Block<T>& block, const char *name);
            bool checkStash(int nodeId, const char *name);

            //nd4j::NDArray<T>* extractArray(nd4j::graph::Block<T>& block, const char *name);
            nd4j::NDArray<T>* extractArray(int nodeId, const char *name);

            void clear();
        };
    }
}




#endif //LIBND4J_STASH_H
