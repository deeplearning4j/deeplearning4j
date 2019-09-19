/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author raver119@gmail.com
//

#include <samediff/samediff_c.h>
#include <samediff/SameDiff.h>
#include <samediff/NodeArgs.h>
#include <ops/declarable/OpRegistrator.h>

void*  SD_createGraph() {
    return new samediff::SameDiff();
}

void  SD_destroyGraph(void *sd) {
    delete reinterpret_cast<samediff::SameDiff*>(sd);
}

void* SD_addVariableNode(void *vsd, void* varray, const char *name) {
    auto array = reinterpret_cast<nd4j::NDArray*>(varray);
    auto sd = reinterpret_cast<samediff::SameDiff*>(vsd);
    auto var = new nd4j::graph::Variable(array, name);
    return sd->graph()->addVariableNode(var);
}

void  SD_execute(void *vsd) {
    auto sd = reinterpret_cast<samediff::SameDiff*>(vsd);
    sd->execute();
}

void  SD_executePartially(void *sd, const char *nodeName) {

}

// args entity: CRUD
void* SD_createArgs() {
    return new samediff::NodeArgs();
}

void  SD_destroyArgs(void* args) {
    delete reinterpret_cast<samediff::NodeArgs*>(args);
}

void  SD_addInputArg(void *args, int position, int nodeId, int index) {
    reinterpret_cast<samediff::NodeArgs*>(args)->addInput(position, nodeId, index);
}

void  SD_addIArg(void *args, int position, Nd4jLong arg) {
    reinterpret_cast<samediff::NodeArgs*>(args)->addIArg(position, arg);
}

void  SD_addDArg(void *args, int position, double arg) {
    reinterpret_cast<samediff::NodeArgs*>(args)->addTArg(position, arg);
}

void  SD_addBArg(void *args, int position, bool arg) {
    reinterpret_cast<samediff::NodeArgs*>(args)->addBArg(position, arg);
}

// node entity: CRUD
void* SD_createNode(void* vsd, const char* opName, void *varg, const char* nodeName) {
    auto sd = reinterpret_cast<samediff::SameDiff*>(vsd);
    auto args = reinterpret_cast<samediff::NodeArgs*>(varg);
    auto node = new nd4j::graph::Node(nd4j::ops::OpRegistrator::getInstance()->getOperation(opName), sd->graph()->nextNodeId(), args->inputs(), {}, {}, 0.0f, args->targs(), args->iargs(), args->bargs());

    if (nodeName != nullptr)
        node->setName(std::string(nodeName));

    sd->graph()->addNode(node);
    return node;
}

int   SD_nodeId(void *node) {
    return reinterpret_cast<nd4j::graph::Node*>(node)->id();
}

// variable entity: CRUD
void* SD_createVariable(void* sd) {
    return nullptr;
}

int   SD_variableId(void *variable) {
    return reinterpret_cast<samediff::Variable*>(variable)->nodeId();
}

int   SD_variableIndex(void *variable) {
    return reinterpret_cast<samediff::Variable*>(variable)->index().second;
}

void* SD_variableById(void *vsd, int nodeId, int index) {
    auto sd = reinterpret_cast<samediff::SameDiff*>(vsd);
    return sd->graph()->getVariableSpace()->getVariable(nodeId, index)->getNDArray()->dup();
}


// tuple entity: CRUD
void* SD_createTuple() {
    return nullptr;
}
