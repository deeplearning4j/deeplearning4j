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
            VariableSpace<float> *_variableSpace;

            // vector holds ID's of top nodes only
            std::vector<int32_t > *_nodes;
            std::map<int32_t, nd4j::graph::Node *> *_mapped;

            std::map<int, std::vector<nd4j::graph::Node *> *> *_onion;
            std::map<int32_t, nd4j::graph::Node *> _unmapped;

            std::mutex _mutexPreprocessing;
            std::atomic<bool> _built;

            std::vector<NDArray<float> *> _output;

            Nd4jStatus validateNode(nd4j::graph::Node *node);

            void expandOnion(int newLayer);

            void injectNode(nd4j::graph::Node *node);
        public:
            Graph(const FlatGraph *flatGraph = nullptr);

            ~Graph();

            // method that'll print out graph
            Nd4jStatus validate();

            // this method will build structured representation of graph
            Nd4jStatus buildGraph();

            int rootNodes();
            int totalNodes();

            nd4j::graph::VariableSpace<float> *getVariableSpace();

            void addNode(nd4j::graph::Node *node);

            std::map<int, std::vector<nd4j::graph::Node *> *> *getOnion();
            std::map<int32_t, nd4j::graph::Node *> *getMapped();

            std::vector<NDArray<float> *> *fetchOutputs();
        };
    }
}

std::vector<NDArray<float> *> * nd4j::graph::Graph::fetchOutputs() {
    return &_output;
}

std::map<int32_t, nd4j::graph::Node *> * nd4j::graph::Graph::getMapped() {
    return _mapped;
};

std::map<int, std::vector<nd4j::graph::Node *> *>* nd4j::graph::Graph::getOnion() {
    return _onion;
}

void nd4j::graph::Graph::injectNode(nd4j::graph::Node *node) {
    if (node->getLayer() < 0)
        throw std::runtime_error("Only nodes with non-negative layer defined can be inserted");

    printf("Node_%i mapped to layer_%i\n", node->id(), node->getLayer());
    fflush(stdout);

    std::pair<int32_t, nd4j::graph::Node *> pair(node->id(), node);
    _onion->at(node->getLayer())->push_back(node);
    _mapped->insert(pair);
}

void nd4j::graph::Graph::expandOnion(int newLayer) {
    std::vector<nd4j::graph::Node *> *rootList = new std::vector<nd4j::graph::Node *>();
    std::pair<int, std::vector<nd4j::graph::Node *>*> pair(newLayer, rootList);
    _onion->insert(pair);
}

nd4j::graph::VariableSpace<float> * nd4j::graph::Graph::getVariableSpace() {
    return _variableSpace;
}

nd4j::graph::Graph::~Graph() {
    delete _mapped;
    delete _nodes;
    delete _variableSpace;
    delete _onion;

    // delete _onion content here
}

void nd4j::graph::Graph::addNode(nd4j::graph::Node *node) {
    _built.store(false);

    std::pair<int32_t, nd4j::graph::Node *> pair(node->id(), node);
    // if model has only external variables as input - it goes to first layer, no matter what.
    if (node->hasExternalInputs() && !node->hasInternalInputs()) {
        node->setLayer(0);

        _onion->at(0)->push_back(node);
        _mapped->insert(pair);

        printf("Node_%i mapped to layer_%i\n", node->id(), node->getLayer());
        fflush(stdout);
    } else {
        // in some cases we're able to put stuff immediately
        if (node->hasInternalInputs() && !node->hasExternalInputs() && node->input()->size() == 1) {

            // we only can put single input nodes, whose outputs were not mapped yet
            if (_mapped->count(node->input()->at(0)) == 1 && (node->output()->size() == 0 || _mapped->count(node->output()->at(0)) == 0)) {
                auto parent = _mapped->at(node->input()->at(0));
                int nLayer = parent->getLayer() + 1;
                if (_onion->count(nLayer) != 1) {
                    expandOnion(nLayer);
                }

                node->setLayer(nLayer);
                _onion->at(nLayer)->push_back(node);
                _mapped->insert(pair);
                printf("Node_%i mapped to layer_%i\n", node->id(), node->getLayer());
                fflush(stdout);

                return;
            }
        }

        // otherwise we're putting it to unmapped space for further sorting
        _unmapped.insert(pair);
    }
}

Nd4jStatus nd4j::graph::Graph::buildGraph() {
    while (_unmapped.size() > 0) {

        // first pass for unmapped nodes, we try to build tale here
        std::map<int32_t, nd4j::graph::Node *>::iterator it;
        for ( it = _unmapped.begin(); it != _unmapped.end(); it++ ) {
            auto node = it->second;

            // single-input node
            if (node->input()->size() == 1) {

                printf("Trying SI Node_%i\n", node->id());
                fflush(stdout);

                int iNode = node->input()->at(0);
                if (_mapped->count(iNode) > 0) {
                    int maxLayer = _mapped->at(iNode)->getLayer() + 1;

                    node->setLayer(maxLayer);
                    if (_onion->count(maxLayer) == 0)
                        expandOnion(maxLayer);

                    this->injectNode(node);
                } else
                    continue;

                _unmapped.erase(node->id());
            } else {
                // multi-input node
                printf("Trying MI Node_%i\n", node->id());
                fflush(stdout);

                int maxLayer = 0;
                for (int e = 0; e < node->input()->size(); e++) {
                    int nodeId = node->input()->at(e);

                    // if input node wasn't mapped yet - we'll have skip it in this round
                    if (_mapped->count(nodeId) == 1) {
                        auto iNode = _mapped->at(nodeId);

                        if (maxLayer < iNode->getLayer())
                            maxLayer = iNode->getLayer();
                    } else
                        continue;
                }

                maxLayer++;
                if (_onion->count(maxLayer) == 0)
                    expandOnion(maxLayer);

                node->setLayer(maxLayer);
                injectNode(node);

                _unmapped.erase(node->id());
            }
        }


        // second pass is mover, we'll be moving onion layers around here
    }

    if (_unmapped.size() == 0)
        _built.store(true);
}

nd4j::graph::Graph::Graph(const FlatGraph *flatGraph) {
    this->_onion = new std::map<int, std::vector<nd4j::graph::Node *> *>();
    this->_mapped = new std::map<int32_t, nd4j::graph::Node *> ();
    this->_nodes = new std::vector<int32_t>();
    this->_variableSpace = new VariableSpace<float>();

    // add 0 layer
    this->expandOnion(0);


    // rolling through nodes
    if (flatGraph != nullptr && flatGraph->nodes() != nullptr && flatGraph->nodes()->size() > 0) {

        // flag to be raised if there's nodes without output being set
        bool outputPassNeeded = false;

        for (int e = 0; e < flatGraph->nodes()->size(); e++) {
            auto node = flatGraph->nodes()->Get(e);

            if (node->output() == nullptr || node->output()->size() == 0) {
                outputPassNeeded = true;
                printf("Orphan node detected: %i\n", node->id());
            }

            this->addNode(new Node(node));
        }
    }

    // parsing variables here
    if (flatGraph != nullptr && flatGraph->variables() != nullptr && flatGraph->variables()->size() > 0) {
        for (int e = 0; e < flatGraph->variables()->size(); e++) {
            auto flatVar = flatGraph->variables()->Get(e);

            auto var = new Variable<float>(flatVar);
            _variableSpace->putVariable(flatVar->id(), var);
        }
    }
}


/**
 * This method returns number of root nodes in this graph
 * @return
 */
int nd4j::graph::Graph::rootNodes() {
    return this->_onion->at(0)->size();
}

/**
 * This method returns total number of nodes in this graph
 * @return
 */
int nd4j::graph::Graph::totalNodes() {
    if (_built != true)
        buildGraph();

    return _mapped->size();
}


Nd4jStatus nd4j::graph::Graph::validate() {
    if (!_built) {
        _mutexPreprocessing.lock();
        if (!_built) {
            this->buildGraph();
        }
        _mutexPreprocessing.unlock();
    }

    if (_built != true)
        return ND4J_STATUS_BAD_GRAPH;

    return ND4J_STATUS_OK;
};

Nd4jStatus nd4j::graph::Graph::validateNode(nd4j::graph::Node *node) {
    // TODO: to be implemented
}

#endif //LIBND4J_GRAPH_H
