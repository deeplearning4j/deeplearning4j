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

            Nd4jStatus executeFlatNode(nd4j::graph::Node *node);
            Nd4jStatus validateNode(nd4j::graph::Node *node);
        public:
            Graph(const FlatGraph *flatGraph = nullptr);

            ~Graph();

            // this method executes graph with current state of VariableSpace
            Nd4jStatus execute();

            // method that'll print out graph
            Nd4jStatus validate();

            int rootNodes();
            int totalNodes();

            nd4j::graph::VariableSpace<float> *getVariableSpace();

            void addNode(Node *node);
        };
    }
}


nd4j::graph::VariableSpace<float> * nd4j::graph::Graph::getVariableSpace() {
    return _variableSpace;
}

nd4j::graph::Graph::~Graph() {
    delete _mapped;
    delete _nodes;
    delete _variableSpace;
}

void nd4j::graph::Graph::addNode(Node *node) {
    // checking for root node
    if (node->input() == nullptr || (node->input()->size() == 1 && node->input()->at(0) < 0)) {
        _nodes->push_back(node->id());
    }

    std::pair<int32_t, nd4j::graph::Node *> pair(node->id(), node);
    _mapped->insert(pair);
}

nd4j::graph::Graph::Graph(const FlatGraph *flatGraph) {
    this->_mapped = new std::map<int32_t, nd4j::graph::Node *> ();
    this->_nodes = new std::vector<int32_t>();
    this->_variableSpace = new VariableSpace<float>();

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

        if (outputPassNeeded) {
            for (int e = 0; e < flatGraph->nodes()->size(); e++) {

            }
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

    // FIXME: this is bad!!!11oneoneleven
    std::map<int32_t, nd4j::graph::Node *>::iterator it;
    for ( it = _mapped->begin(); it != _mapped->end(); it++ ) {
                   it->second->prepare();
    }

#pragma omp parallel for if (_nodes->size()>1) schedule(guided) proc_bind(spread)
//#pragma omp parallel for schedule(dynamic) proc_bind(spread)
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

    return ND4J_STATUS_OK;
}

#endif //LIBND4J_GRAPH_H
