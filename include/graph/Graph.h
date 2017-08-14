//
// @author raver119@gmail.com
//

#ifndef LIBND4J_GRAPH_H
#define LIBND4J_GRAPH_H

#include <list>
#include <map>
#include <NDArray.h>
#include <graph/Node.h>
#include <graph/VariableSpace.h>
#include <graph/generated/node_generated.h>
#include <graph/generated/graph_generated.h>

namespace nd4j {
    namespace graph {
        class Graph {
        protected:
            // vector holds ID's of top nodes only
            std::vector<int32_t > *_nodes;

            std::map<int32_t, const nd4j::graph::FlatNode *> *_mapped;
            const FlatGraph *_flatGraph;

            Nd4jStatus executeFlatNode(const nd4j::graph::FlatNode *node);

            Nd4jStatus validateNode(const FlatNode *node);
        public:
            Graph(const FlatGraph *flatGraph = nullptr);

            ~Graph();

            Nd4jStatus execute();

            // method that'll print out graph
            Nd4jStatus validate();

            int rootNodes();

            int totalNodes();
        };
    }
}

nd4j::graph::Graph::~Graph() {
    delete _mapped;
    delete _nodes;
}

nd4j::graph::Graph::Graph(const FlatGraph *flatGraph) {
    this->_flatGraph = flatGraph;
    this->_mapped = new std::map<int32_t, const nd4j::graph::FlatNode *> ();
    this->_nodes = new std::vector<int32_t>();

    if (this->_flatGraph != nullptr && this->_flatGraph->nodes()->size() > 0) {

        // flag to be raised if there's nodes without output being set
        bool outputPassNeeded = false;

        for (int e = 0; e < this->_flatGraph->nodes()->size(); e++) {
            auto node = this->_flatGraph->nodes()->Get(e);

            // checking for root node
            if (node->input() == nullptr || (node->input()->size() == 1 && node->input()->Get(0) == 0)) {
                _nodes->push_back(node->id());
            }

            if (node->output() == nullptr || node->output()->size() == 0) {
                outputPassNeeded = true;
                printf("Orphan node detected: %i\n", node->id());
            }

            std::pair<int32_t, const nd4j::graph::FlatNode*> pair(node->id(), node);
            _mapped->insert(pair);
        }

        if (outputPassNeeded) {
            for (int e = 0; e < this->_flatGraph->nodes()->size(); e++) {

            }
        }
    }
}

/**
 * This method executes this graph and all it subgraphs
 * @return
 */
Nd4jStatus nd4j::graph::Graph::execute() {
    for (int e = 0; e < _nodes->size(); e++) {
        auto n = _nodes->at(e);

        executeFlatNode(_mapped->at(n));
    }

    return ND4J_STATUS_OK;
}

/**
 * This method returns number of root nodes in this graph
 * @return
 */
int nd4j::graph::Graph::rootNodes() {
    return this->_nodes->size();
}

/**
 * This method returns total number of nodes in this graph
 * @return
 */
int nd4j::graph::Graph::totalNodes() {
    return _mapped->size();
}


Nd4jStatus nd4j::graph::Graph::validate() {
/*    for (auto n: this->_nodes->begin()) {
        printf("Node Name: %s\n", n->name()->c_str());

        validateNode(n);
    }
    */

    return ND4J_STATUS_OK;
};

Nd4jStatus nd4j::graph::Graph::validateNode(const FlatNode *node) {

    if (node->input()->size() > 0) {
        for (int e = 0; e < node->input()->size(); e++) {

            // we want to ensure that input<->output dimensionality matches here, and TADs look reasonable
        }
    }

    if (node->output()->size() > 0) {
        for (int e = 0; e < node->output()->size(); e++) {
            int n = node->output()->Get(e);

            // output can be either id of other node, or -1 in case of variable output
            if (n >= 0 && _mapped->count(n) == 0)
                throw std::invalid_argument("Bad output node");

            // we also want to see data type here to match the next node
        }
    }


}

Nd4jStatus nd4j::graph::Graph::executeFlatNode(const nd4j::graph::FlatNode *node) {
    printf("Executing node_%i\n", node->id());

    // TODO: put execution code here


    // going down to next node here
    if (node->output() != nullptr && node->output()->size() > 0) {
        for (int e = 0; e < node->output()->size(); e++) {
            auto n = node->output()->Get(e);

            // we skip non-positive values here
            if (n > 0)
                executeFlatNode(_mapped->at(n));
        }
    }

    return ND4J_STATUS_OK;
}

#endif //LIBND4J_GRAPH_H
