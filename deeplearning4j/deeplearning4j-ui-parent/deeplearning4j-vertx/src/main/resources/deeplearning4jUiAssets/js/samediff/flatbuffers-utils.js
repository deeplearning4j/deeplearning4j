/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */


function extractHeaders(/*Uint8Array*/ bytes, offset){
    var header1a = bytes.slice(offset+0,offset+4);
    var header1b = bytes.slice(offset+4,offset+8);
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

//Return graph inputs as a String[]
function uiGraphGetInputs(/*UIGraphStructure*/ graph){
    var inLength = graph.inputsLength();
    var inputs = [];
    for( var i=0; i<inLength; i++ ){
        inputs.push(graph.inputs(i));
    }
    return inputs;
}

//Return graph outputs as a String[]
function uiGraphGetOutputs(/*UIGraphStructure*/ graph){
    var inLength = graph.outputsLength();
    var outputs = [];
    for( var i=0; i<inLength; i++ ){
        outputs.push(graph.outputs(i));
    }
    return outputs;
}

//Return graph variables as nd4j.graph.UIVariable[]
function uiGraphGetVariables(/*UIGraphStructure*/ graph){
    var varsLength = graph.variablesLength();
    var vars = [];
    for( var i=0; i<varsLength; i++ ){
        vars.push(graph.variables(i));
    }
    return vars;
}

//Return graph variables as String[]
function uiGraphGetVariableNames(/*UIGraphStructure*/ graph){
    var varsLength = graph.variablesLength();
    var vars = [];
    for( var i=0; i<varsLength; i++ ){
        vars.push(graph.variables(i).name());
    }
    return vars;
}

//Returns nd4j.graph.UIOp[]
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
        case nd4j.graph.VarType.ARRAY:
            return "array";
        default:
            return "" + varType;
    }
}

function varShapeToString(/*UIVariable*/ uivar){
    var n = uivar.shapeLength();
    if(n === 0)
        return "";
    var shape = [];
    for( var i=0; i<n; i++ ){
        var l = uivar.shape(i);
        var s = l.toString();
        var s2 = l.toFloat64().toString();
        shape.push(s2);
    }
    return "[" + shape.toString() + "]";
}

function dataTypeToString(dataTypeByte){
    switch (dataTypeByte){
        case nd4j.graph.DataType.INHERIT:
            return "INHERIT";
        case nd4j.graph.DataType.BOOL:
            return "BOOL";
        case nd4j.graph.DataType.FLOAT8:
            return "FLOAT8";
        case nd4j.graph.DataType.HALF:
            return "HALF";
        case nd4j.graph.DataType.HALF2:
            return "HALF2";
        case nd4j.graph.DataType.FLOAT:
            return "FLOAT";
        case nd4j.graph.DataType.DOUBLE:
            return "DOUBLE";
        case nd4j.graph.DataType.INT8:
            return "INT8";
        case nd4j.graph.DataType.INT16:
            return "INT16";
        case nd4j.graph.DataType.INT32:
            return "INT32";
        case nd4j.graph.DataType.INT64:
            return "INT64";
        case nd4j.graph.DataType.UINT8:
            return "UINT8";
        case nd4j.graph.DataType.UINT16:
            return "UINT16";
        case nd4j.graph.DataType.UINT32:
            return "UINT32";
        case nd4j.graph.DataType.UINT64:
            return "UINT64";
        case nd4j.graph.DataType.QINT8:
            return "QINT8";
        case nd4j.graph.DataType.QINT16:
            return "QINT16";
        case nd4j.graph.DataType.BFLOAT16:
            return "BFLOAT16";
        case nd4j.graph.DataType.UTF8:
            return "UTF8";
        default:
            return "" + dataTypeByte;
    }
}

function dataTypeBytesPerElement(dataTypeByte){
    switch (dataTypeByte){
        case nd4j.graph.DataType.BOOL:
        case nd4j.graph.DataType.FLOAT8:
        case nd4j.graph.DataType.INT8:
        case nd4j.graph.DataType.UINT8:
        case nd4j.graph.DataType.QINT8:
            return 1;
        case nd4j.graph.DataType.HALF:
        case nd4j.graph.DataType.HALF2:
        case nd4j.graph.DataType.INT16:
        case nd4j.graph.DataType.UINT16:
        case nd4j.graph.DataType.QINT16:
        case nd4j.graph.DataType.BFLOAT16:
            return 2;
        case nd4j.graph.DataType.FLOAT:
        case nd4j.graph.DataType.INT32:
        case nd4j.graph.DataType.UINT32:
            return 4;
        case nd4j.graph.DataType.DOUBLE:
        case nd4j.graph.DataType.INT64:
        case nd4j.graph.DataType.UINT64:
            return 8;
        case nd4j.graph.DataType.UTF8:
            return 0;   //TODO
        default:
            return "" + dataTypeByte;
    }
}

function getScalar(/*FlatArray*/ flatArray, /*number[]*/ idxs){
    //First: work out offset... assume C order here
    var offset = 0;
    // var prod = 1;
    var rank = flatArray.shape(0).toFloat64();  //Note: shape is in nd4j format. So rank, shape. We're assuming C order here. Note also shape(i) returns flatbuffers long object
    for( var i=0; i<rank; i++ ){
        var size = flatArray.shape(i+1).toFloat64();
        var stride = flatArray.shape(rank+i+1).toFloat64();
        offset += stride * idxs[i];
        // prod *= size.toFloat64();
    }
    return scalarFromFlatArrayIdx(flatArray, offset);
}

function scalarFromFlatArray(/*FlatArray*/ flatArray) {
    return scalarFromFlatArrayIdx(flatArray, 0);
}

function scalarFromFlatArrayIdx(/*FlatArray*/ flatArray, idx){
    //TODO OPTIMIZE THIS!
    var dt = flatArray.dtype();
    switch (dt){
        //Skip hard to decode types for now
        case nd4j.graph.DataType.FLOAT8:
        case nd4j.graph.DataType.QINT8:
        case nd4j.graph.DataType.HALF:
        case nd4j.graph.DataType.HALF2:
        case nd4j.graph.DataType.QINT16:
        case nd4j.graph.DataType.BFLOAT16:
            return null;
    }

    var bytesPerElem = dataTypeBytesPerElement(dt);
    // var array = new Uint8Array(numBytes);
    var dv = new DataView(new ArrayBuffer(bytesPerElem));
    var byteOffset = idx * bytesPerElem;
    var j=0;
    for(var i=byteOffset; i<byteOffset + bytesPerElem; i++ ){
        var signedByte = flatArray.buffer(i);
        // array[i] = signedByte;      //TODO do we need to convert here???
        dv.setInt8(j++, signedByte);
    }

    //Note: "get(idx)" is byte offset
    var out;
    switch (dt){
        case nd4j.graph.DataType.BOOL:
            out = (dv.getUint8(0) === 0 ? "false" : "true");
            break;
        case nd4j.graph.DataType.INT8:
            out = dv.getInt8(0);
            break;
        case nd4j.graph.DataType.UINT8:
            out = dv.getUint8(0);
            break;
        case nd4j.graph.DataType.INT16:
            out = dv.getInt16(0);
            break;
        case nd4j.graph.DataType.UINT16:
            out = dv.getUint16(0);
            break;
        case nd4j.graph.DataType.INT32:
            out = dv.getInt32(0);
            break;
        case nd4j.graph.DataType.UINT32:
            out = dv.getUint32(0);
            break;
        case nd4j.graph.DataType.INT64:
            //No getInt64 method... :/
            //TODO Need a solution to this!
            out = "<int64>";
            break;
        case nd4j.graph.DataType.UINT64:
            out = "<uint64>";
            break;
        case nd4j.graph.DataType.FLOAT:
            out = dv.getFloat32(0);
            break;
        case nd4j.graph.DataType.DOUBLE:
            out = dv.getFloat64(0);
            break;
        case nd4j.graph.DataType.UTF8:
            //TODO need to decode from bytes here
            out = "<utf8>";
            break;
        default:
            return "";
    }
    return out;
}

