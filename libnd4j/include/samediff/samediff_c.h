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


#ifndef SAMEDIFF_SAMEDIFF_C_H
#define SAMEDIFF_SAMEDIFF_C_H

#include <pointercast.h>
#include <dll.h>

extern "C" {
    // graph entity: CRUD
    ND4J_EXPORT void* SD_createGraph();
    ND4J_EXPORT void  SD_destroyGraph(void *sd);
    ND4J_EXPORT void  SD_execute(void *sd);
    ND4J_EXPORT void  SD_executePartially(void *sd, const char *nodeName);
    ND4J_EXPORT void* SD_addVariableNode(void *sd, void* array, const char *name);

    // args entity: CRUD
    ND4J_EXPORT void* SD_createArgs();
    ND4J_EXPORT void  SD_destroyArgs(void* args);
    ND4J_EXPORT void  SD_addInputArg(void *args, int position, int nodeId, int index);
    ND4J_EXPORT void  SD_addIArg(void *args, int position, Nd4jLong arg);
    ND4J_EXPORT void  SD_addDArg(void *args, int position, double arg);
    ND4J_EXPORT void  SD_addBArg(void *args, int position, bool arg);

    // node entity: CRUD
    ND4J_EXPORT void* SD_createNode(void* sd, const char* opName, void *arg, const char* nodeName);
    ND4J_EXPORT int   SD_nodeId(void *node);


    // variable entity: CRUD
    ND4J_EXPORT void* SD_createVariable(void* sd);
    ND4J_EXPORT int   SD_variableId(void *variable);
    ND4J_EXPORT int   SD_variableIndex(void *variable);
    ND4J_EXPORT void* SD_variableById(void *sd, int nodeId, int index);

    // tuple entity: CRUD
    ND4J_EXPORT void* SD_createTuple();
};

#endif //SAMEDIFF_SAMEDIFF_C_H
