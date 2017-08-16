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

            Nd4jStatus executeFlatNode(nd4j::graph::Node *node);
            Nd4jStatus validateNode(nd4j::graph::Node *node);

            void expandOnion(int newLayer);


            void injectNode(nd4j::graph::Node *node);

        public:
            Graph(const FlatGraph *flatGraph = nullptr);

            ~Graph();

            // this method executes graph with current state of VariableSpace
            Nd4jStatus execute();

            // method that'll print out graph
            Nd4jStatus validate();

            // this method will build structured representation of graph
            Nd4jStatus buildGraph();

            int rootNodes();
            int totalNodes();

            nd4j::graph::VariableSpace<float> *getVariableSpace();

            void addNode(nd4j::graph::Node *node);

        };
    }
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

                if (node->output()->size() <= 1) {
                    // single-output node

                    // in this case we just move this one to the bottom
                    //if (node->output()->size() == 0 || (node->hasExternalOutputs() && !node->hasInternalOutputs())) {

                        int iNode = node->input()->at(0);
                        if (_mapped->count(iNode) > 0) {
                            int maxLayer = _mapped->at(iNode)->getLayer();

                            node->setLayer(maxLayer);
                            if (_onion->count(maxLayer) == 0)
                                expandOnion(maxLayer);

                            this->injectNode(node);
                        } else
                            continue;
                    //}
                } else {
                    // multi-output node

                    continue;
                }

                _unmapped.erase(node->id());
            } else {
                // multi-input node
                printf("Trying MI Node_%i\n", node->id());

                // single output node
                if (node->output()->size() <= 1) {

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
                } else {
                    // multi-output node

                    continue;
                }

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
 * This method executes this graph and all it subgraphs
 * @return
 */
Nd4jStatus nd4j::graph::Graph::execute() {
    if (!_built) {
        _mutexPreprocessing.lock();
        if (!_built) {
            this->buildGraph();
        }
        _mutexPreprocessing.unlock();
    }

    if (_built != true)
        throw std::runtime_error("Graph wasn't built. Can't execute.");

// we loop through op layers here
    for (int l = 0; l < _onion->size(); l++) {

        for (int n = 0; n < _onion->at(l)->size(); n++) {
            auto node = _onion->at(l)->at(n);

            executeFlatNode(node);
        }
    }

/*
    // FIXME: this is bad!!!11oneoneleven
    std::map<int32_t, nd4j::graph::Node *>::iterator it;
    for ( it = _mapped->begin(); it != _mapped->end(); it++ ) {
                   it->second->prepare();
    }

#pragma omp parallel for if (_nodes->size()>1) num_threads(_nodes->size()) schedule(guided) proc_bind(spread)
//#pragma omp parallel for schedule(dynamic) proc_bind(spread)
    for (int e = 0; e < _nodes->size(); e++) {
        auto n = _nodes->at(e);

        executeFlatNode(_mapped->at(n));
    }
*/
    return ND4J_STATUS_OK;
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
/*    for (auto n: this->_nodes->begin()) {
        printf("Node Name: %s\n", n->name()->c_str());

        validateNode(n);
    }
    */

    return ND4J_STATUS_OK;
};

Nd4jStatus nd4j::graph::Graph::validateNode(nd4j::graph::Node *node) {

    if (node->input()->size() > 0) {
        for (int e = 0; e < node->input()->size(); e++) {

            // we want to ensure that input<->output dimensionality matches here, and TADs look reasonable
        }
    }

    if (node->output()->size() > 0) {
        for (int e = 0; e < node->output()->size(); e++) {
            int n = node->output()->at(e);

            // output can be either id of other node, or -1 in case of variable output
            if (n >= 0 && _mapped->count(n) == 0)
                throw std::invalid_argument("Bad output node");

            // we also want to see data type here to match the next node
        }
    }
}

Nd4jStatus nd4j::graph::Graph::executeFlatNode(nd4j::graph::Node *node) {
    OpType opType = node->opType();
    int opNum = node->opNum();

    //printf("Executing node_%i: opNum: %i; tid: %i;\n", node->id(), opNum, omp_get_thread_num());
    //fflush(stdout);

    // if we have multiple input nodes - we have to wait till input nodes are done
    if (node->isMultiInput()) {
        //printf("Blocking in node_%i\n", node->id());
        //fflush(stdout);
        for (int e = 0; e < node->input()->size(); e++) {
            int in = node->input()->at(e);

            // we don't wait on external variables
            if (in < 0)
                continue;

            _mapped->at(in)->waitTillFinished();
        }
    }

    if (opType == OpType_TRANSFORM) {
        int in = node->input()->at(0);

        auto x = _variableSpace->getVariable(in);

        //printf("Node: %i; Op: %i; BEFORE X: %f\n", node->id(), opNum, x->getNDArray()->getScalar(0));
        //fflush(stdout);

        // if output of previous node is used in different code branches - duplicate it
        if (in > 0)
            if (_mapped->at(in)->output()->size() > 1) {
                auto array = x->getNDArray()->dup(x->getNDArray()->ordering());
                x = new Variable<float>(array);
            };

        functions::transform::Transform<float>::template exec(opNum, x->getNDArray()->_buffer,
                                                                  x->getNDArray()->_shapeInfo,
                                                                  x->getNDArray()->_buffer,
                                                                  x->getNDArray()->_shapeInfo, node->extraParams(), nullptr,
                                                                  nullptr);

        _variableSpace->putVariable(node->id(), x);

        //printf("Node: %i; Op: %i; AFTER X: %f\n", node->id(), opNum, x->getNDArray()->getScalar(0));
        //fflush(stdout);

        if (node->hasExternalOutputs()) {
            for (int e = 0; e < node->output()->size(); e++) {
                if (node->output()->at(e) > 0)
                    continue;

                auto out = _variableSpace->getVariable(node->output()->at(e));

                // assign output
                if (out->getNDArray() != x->getNDArray())
                    out->getNDArray()->assign(x->getNDArray());
            }
        }
    } else if (opType == OpType_PAIRWISE) {

        //printf("PWT> x: %i; y: %i\n", node->input()->at(0), node->input()->at(1));
        //fflush(stdout);

        auto x = _variableSpace->getVariable(node->input()->at(0));
        auto y = _variableSpace->getVariable(node->input()->at(1));

        //printf("PWT> X: %f; Y: %f\n", x->getNDArray()->getScalar(0), y->getNDArray()->getScalar(0));
        //fflush(stdout);

        auto z = x;
        if (node->output()->size() > 0) {
            z = new Variable<float>(new NDArray<float>(x->getNDArray()));
        }


        functions::pairwise_transforms::PairWiseTransform<float>:: template exec(opNum, x->getNDArray()->_buffer, x->getNDArray()->_shapeInfo, y->getNDArray()->_buffer, y->getNDArray()->_shapeInfo,
                                                                                 z->getNDArray()->_buffer, z->getNDArray()->_shapeInfo, node->extraParams());

        _variableSpace->putVariable(node->id(), z);


        //printf("PWT> z: %f;\n", z->getNDArray()->getScalar(0));
        //fflush(stdout);

        if (node->hasExternalOutputs()) {
            for (int e = 0; e < node->output()->size(); e++) {
                if (node->output()->at(e) > 0)
                    continue;

                auto out = _variableSpace->getVariable(node->output()->at(e));

                // assign output
                if (out->getNDArray() != z->getNDArray())
                    out->getNDArray()->assign(z->getNDArray());
            }
        }
    }

    node->finished();

    /*
    // going down to next node here
    if (node->output() != nullptr && node->output()->size() > 0) {

        // if next node is multi-output, only 0 thread goes in
        //if (!node->isMultiInput() || omp_get_thread_num() == 0) {
            int s = node->output()->size();
//#pragma omp parallel for if (s>1) schedule(dynamic, 1) proc_bind(spread)
            for (int e = 0; e < s; e++) {
                auto n = node->output()->at(e);

                // we skip non-positive values here
                if (n != 0 && _mapped->count(n) > 0) {
                    auto nextNode = _mapped->at(n);
                  //  printf("Op: %i; N: %i; S: %i\n", nextNode->opNum(), omp_get_thread_num(), s);

                    // last input node continues here
                    bool m = false;
                    if (nextNode->isMultiInput())
                        m = nextNode->input()->at(nextNode->input()->size()-1) == node->id();


                    // only tid_0 invokes multi-input node, block will happen right there
                    if (nextNode->isMultiInput() && m )
                        continue;
                    else
                        executeFlatNode(nextNode);
                } // else
                  //  printf("Skipping node_%i\n", n);
            }
       // }
    }
    */

    return ND4J_STATUS_OK;
}

#endif //LIBND4J_GRAPH_H
