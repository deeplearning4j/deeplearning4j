/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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



function extractHeaders(/*Uint8Array*/ bytes){
    var header1a = bytes.slice(0,4);
    var header1b = bytes.slice(4,8);
    var headerLength = byteArrayToInt(header1a);
    var contentLength = byteArrayToInt(header1b);
    return [headerLength, contentLength];
}


function byteArrayToInt(byteArray) {
    var value = 0;
    for ( var i = 0; i < byteArray.length; i++) {
        value = (value * 256) + byteArray[i];
    }
    return value;
}

function decodeStaticInfo(headerContentBytes, bufferContentBytes){

    var headerBuffer = new flatbuffers.ByteBuffer(headerContentBytes);
    var contentBuffer = new flatbuffers.ByteBuffer(bufferContentBytes);


    var header = nd4j.graph.UIStaticInfoRecord.getRootAsUIStaticInfoRecord(headerBuffer);
    var infoType = header.infoType();
    switch(infoType){
        case nd4j.graph.UIInfoType.GRAPH_STRUCTURE:
            var graphStructure = nd4j.graph.UIGraphStructure.getRootAsUIGraphStructure(contentBuffer);
            return ["graph", graphStructure];
        case nd4j.graph.UIInfoType.SYTEM_INFO:
            var info = nd4j.graph.UISystemInfo.getRootAsUISystemInfo(contentBuffer);
            return ["systeminfo", info];
        case nd4j.graph.UIInfoType.START_EVENTS:
            return ["startevents", null];
        default:
            console.log("Unknown static information type: " + infoType);
            return null;

    }
}

function uiGraphGetInputs(/*UIGraphStructure*/ graph){
    var inLength = graph.inputsLength();
    var inputs = [];
    for( var i=0; i<inLength; i++ ){
        inputs.push(graph.inputs(i));
    }
    return inputs;
}

function uiGraphGetOutputs(/*UIGraphStructure*/ graph){
    var inLength = graph.outputsLength();
    var outputs = [];
    for( var i=0; i<inLength; i++ ){
        outputs.push(graph.outputs(i));
    }
    return outputs;
}

function uiGraphGetVariables(/*UIGraphStructure*/ graph){
    var varsLength = graph.variablesLength();
    var vars = [];
    for( var i=0; i<varsLength; i++ ){
        vars.push(graph.variables(i));
    }
    return vars;
}

function uiGraphGetOps(/*UIGraphStructure*/ graph){
    var opsLength = graph.opsLength();
    var ops = [];
    for( var i=0; i<opsLength; i++ ){
        ops.push(graph.ops(i));
    }
    return ops;
}

function varTypeToString(varType){
    switch (varType){
        case nd4j.graph.VarType.CONSTANT:
            return "constant";
        case nd4j.graph.VarType.PLACEHOLDER:
            return "placeholder";
        case nd4j.graph.VarType.VARIABLE:
            return "variable";
        default:
            return "" + varType;
    }
}

