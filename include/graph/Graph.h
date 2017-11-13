//
// @author raver119@gmail.com
//

#ifndef LIBND4J_GRAPH_H
#define LIBND4J_GRAPH_H

#include <list>
#include <algorithm>
#include <map>
//#include <NDArray.h>
#include <graph/Node.h>
#include <graph/Stash.h>
#include <Scope.h>
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <graph/generated/node_generated.h>
#include <graph/generated/graph_generated.h>
#include <graph/generated/config_generated.h>
#include <ExecutorConfiguration.h>

namespace nd4j {
    namespace graph {

        template <typename T>
        class Graph {
        protected:
            ExecutorConfiguration *_configuration;
            VariableSpace<T> *_variableSpace;
            Stash<T>* _stash;

            // this list holds references to Node ptrs, which should be free'd in Graph destructor
            std::vector<Node<T> *> _handles;

            // vector holds ID's of top nodes only
            std::vector<int32_t > *_nodes;
            std::map<int32_t, nd4j::graph::Node<T> *> *_mapped;

            std::map<int, std::vector<nd4j::graph::Node<T> *> *> *_onion;
            std::map<int32_t, nd4j::graph::Node<T> *> _unmapped;

            std::mutex _mutexPreprocessing;
            std::atomic<bool> _built;

            std::vector<int32_t> _output;
            std::vector<int32_t> _autos;


            std::map<int, Scope<T> *> _mappedScopes;
            std::vector<Scope<T> *> _scopes;

////////////////////////////////////////
            Nd4jStatus validateNode(nd4j::graph::Node<T> *node);

            void expandOnion(int newLayer);

            void injectNode(nd4j::graph::Node<T> *node);

            void pushToOutputOnce(int32_t id);

            void printOutNode(Node<T>* node);
        public:
            Graph(const FlatGraph *flatGraph = nullptr);

            ~Graph();

            // method that'll print out graph
            Nd4jStatus validate();

            // this method will build structured representation of graph
            Nd4jStatus buildGraph();

            // this method will return estimated memory size (in bytes) required for 1 full graph execution round
            Nd4jIndex estimateRequiredMemory();

            // this method returns number of root nodes in this graph
            int rootNodes();

            // this method returns total number of nodes in this graph
            int totalNodes();

            int numberOfPlaceholders();

            std::vector<nd4j::graph::Variable<T>*>* getPlaceholders();

            /**
             * This method returns pointer to thread_local VariableSpace
             * @return
             */
            nd4j::graph::VariableSpace<T> *getVariableSpace();

            /**
             * This method adds given node to the graph
             *
             * @param node
             */
            void addNode(nd4j::graph::Node<T> *node);

            /**
             * This method returns layered representation of the graph
             *
             * @return
             */
            std::map<int, std::vector<nd4j::graph::Node<T> *> *> *getOnion();

            /**
             * This method returns map of all nodes of the graph
             * @return
             */
            std::map<int32_t, nd4j::graph::Node<T> *> *getMapped();

            /**
             * This method returns outputs of of this graph
             * @return
             */
            std::vector<nd4j::graph::Variable<T> *> *fetchOutputs();

            /**
             * This method returns pointer to ExecutorConfiguration
             *
             * @return
             */
            ExecutorConfiguration *getExecutorConfiguration();

            /**
             * This method adds specified node (by ID) to de
             * @param id
             */
            void addOutput(int32_t id);

            /**
             * This method returns all nodes at once (order is NOT guaranteed)
             * @return
             */
            std::vector<nd4j::graph::Node<T>*> *getAllNodes();

            /**
             * This method prints out Graph op-by-op, and respective inputs
             */
            void printOut();

            /**
             * This method returns Scope ptr specified with id
             *
             * @param id
             * @return
             */
            Scope<T>* scopeById(int id);
        };
    }
}

#endif //LIBND4J_GRAPH_H
