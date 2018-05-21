//
// Created by raver119 on 14.10.2017.
//

#ifndef LIBND4J_SCOPE_H
#define LIBND4J_SCOPE_H

#include <string>
#include <map>
#include <graph/Node.h>

namespace nd4j {
    namespace graph {

        /**
         * Scope holds sequential list of operations, and made suitable for continuous
         * re-execution of multiple operations.
         *
         * @tparam T
         */
        template <typename T>
        class Scope {
        protected:
            // Graph-unique IDs for Scope instances
            int _id;
            std::string _name;

            // list of nodes to run, always sequential
            // Graph takes care of topo sort
            std::vector<Node<T> *> _nodes;
        public:
            // attach GiG here, with shared namespace?
            // or just rebuilt graph leaf?
            //          ¯\_(ツ)_/¯

            // default consructor
            explicit Scope(int id, const char* name = nullptr);

            // default destructor
            ~Scope();

            /**
             * this method adds op node to the scope
             *
             * PLEASE NOTE: We assume that ops are being added ORDERED
             */
            void push_back(Node<T>* node);

            /**
             * This method returns list of ops stored earlier, ready for execution
             *
             * PLEASE NOTE: If the scope is conditional - last op in list should be BooleanOp
             * @return
             */
            std::vector<Node<T>*> * nodes();

            /**
             * This function returns number of nodes in this scope
             *
             * @return
             */
            int size();

            /**
             * Returns ID of this scope
             * @return
             */
            int id();

            /**
             * Returns name of this scope
             *
             * @return
             */
            std::string* name();

            /**
             * This method returns clone of this Scope
             */
            Scope<T>* clone();

            template <typename N>
            Scope<N>* asT();

            /**
             * This method removes all Nodes from this scope
             */
            void forgetNodes();
        };
    }
}


#endif //LIBND4J_SCOPE_H
