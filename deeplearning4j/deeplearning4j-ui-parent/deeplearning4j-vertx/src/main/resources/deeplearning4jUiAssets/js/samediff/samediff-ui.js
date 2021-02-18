/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

function toggleSidebar(){
    $('#samediffsidebar').toggleClass('sidebarhidden');
}

var selectedPage = "graph";
function samediffSetPage(pageName){
    $("#sdnavgraph").removeClass("active");
    $("#sdnavplots").removeClass("active");
    $("#sdnaveval").removeClass("active");
    $("#sdnavperf").removeClass("active");

    switch(pageName){
        case "graph":
            $("#sdnavgraph").addClass("active");
            break;
        case "plots":
            $("#sdnavplots").addClass("active");
            break;
        case "evaluation":
            $("#sdnaveval").addClass("active");
            break;
        case "performance":
            $("#sdnavperf").addClass("active");
            break;
    }

    console.log("Selected page: " + pageName);
    selectedPage = pageName;
    renderContent();
}


sdGraphNodes = [];
sdGraphEdges = [];
sdGraphInputs = [];
sdGraphOutputs = [];
sdGraphVariables = [];
sdGraphVariableNames = [];
sdGraphOpsList = [];
sdGraphOpsMap = new Map();
sdGraphVariableMap = new Map();

function fileSelect(evt) {
    var output = [];
    file = evt.target.files[0];
    output.push('<li><strong>', escape(file.name), '</strong> (', file.type || 'n/a', ') - ',
        file.size, ' bytes, last modified: ',
        file.lastModifiedDate ? file.lastModifiedDate.toLocaleDateString() : 'n/a',
        '</li>');
    document.getElementById('selectedfile').innerHTML = "<strong>" + escape(file.name) + "</strong><br>" + file.size + " bytes<br>Modified: " +
        (file.lastModifiedDate ? file.lastModifiedDate.toLocaleDateString() : 'n/a');
    console.log("About to render graph: file " + file.name);


    readGraphStructure();
}

function readGraphStructure(){
    //First: load data
    if (file) {
        var fr = new FileReader();
        var fileData = new Blob([file]);            //TODO Don't load the whole file into memory at once!
        fr.readAsArrayBuffer(fileData);
        fr.onload = function () {
            var arrayBuffer = fr.result;
            var bytes = new Uint8Array(arrayBuffer);
            //console.log(bytes);

            var lengths = extractHeaders(bytes, 0);
            var headerLength = lengths[0];
            var contentLength = lengths[1];

            // console.log("Header/content lengths: " + headerLength + ", " + contentLength);

            //https://gist.github.com/alexvictoor/0b76764857b780ad8f83

            var headerSlice = bytes.slice(8, 8 + headerLength);
            var contentSlice = bytes.slice(8 + headerLength, 8 + headerLength + contentLength);

            var decoded = decodeStaticInfo(headerSlice, contentSlice);

            // console.log("Decoded header message: " + decoded[0]);

            if (decoded[0] === "graph") {
                var opCount = 0;
                var phCount = 0;
                var varCount = 0;
                var constCount = 0;

                sdGraphNodes.length = 0;
                sdGraphEdges.length = 0;
                var graph = decoded[1];
                sdGraphInputs = uiGraphGetInputs(graph);
                sdGraphOutputs = uiGraphGetOutputs(graph);
                sdGraphVariables = uiGraphGetVariables(graph);
                sdGraphVariableNames = uiGraphGetVariableNames(graph);
                sdGraphOpsList = uiGraphGetOps(graph);
                console.log("Inputs: " + sdGraphInputs);
                console.log("Outputs: " + sdGraphOutputs);
                console.log("Variables: " + sdGraphVariableNames);
                console.log("Ops: " + sdGraphOpsList);

                var mapVarNameInteger = new Map();
                sdGraphVariableMap = new Map();            //Key: variable name, value: variable
                var count = 0;
                for (var i = 0; i < sdGraphVariables.length; i++) {
                    var v = sdGraphVariables[i];
                    var name = v.name();
                    mapVarNameInteger.set(count, name);
                    sdGraphVariableMap.set(name, v);
                }

                for (var i = 0; i < sdGraphVariables.length; i++) {
                    var v = sdGraphVariables[i];
                    var name = v.name();
                    //Add variables/constants/placeholders as a node
                    var vType = v.type();

                    switch (vType) {
                        case nd4j.graph.VarType.CONSTANT:
                            constCount++;
                            break;
                        case nd4j.graph.VarType.PLACEHOLDER:
                            phCount++;
                            break;
                        case nd4j.graph.VarType.VARIABLE:
                            varCount++;
                            break;
                    }

                    if (vType === nd4j.graph.VarType.CONSTANT || vType === nd4j.graph.VarType.PLACEHOLDER || vType === nd4j.graph.VarType.VARIABLE) {
                        var dt = dataTypeToString(v.datatype());
                        var shape = varShapeToString(v);
                        var n = "\"" + name + "\"\n" + varTypeToString(vType) + "\n" + dt + " " + shape;

                        var extraLabel = v.uiLabelExtra();
                        if (extraLabel != null && extraLabel !== "") {
                            n = n + "\n" + extraLabel;
                        }


                        if (vType === nd4j.graph.VarType.CONSTANT) {
                            var constArr = v.constantValue();
                            if (constArr != null) {
                                if (constArr.shapeLength() === 0 && constArr.bufferLength() > 0) {
                                    var scalar = scalarFromFlatArray(constArr);
                                    if (scalar != null && scalar !== "") {
                                        n = n + "\nScalar val: " + scalar;
                                    }
                                }
                            }
                        }


                        var nodeObj = {
                            label: n,
                            id: "var-" + name,
                            name: "var-" + name
                        };

                        var renderStyle = "";
                        if (vType === nd4j.graph.VarType.VARIABLE) {
                            renderStyle = "uivariable variable";
                        } else if (vType === nd4j.graph.VarType.PLACEHOLDER) {
                            renderStyle = "uivariable placeholder";
                        } else if (vType === nd4j.graph.VarType.CONSTANT) {
                            renderStyle = "uivariable constant";
                        }

                        sdGraphNodes.push({data: nodeObj, classes: renderStyle});

                        if (v.inputsForOpLength() > 0) {
                            for (var j = 0; j < v.inputsForOpLength(); j++) {
                                var opName = v.inputsForOp(j);
                                opName = idEscapeSlashes(opName);
                                var edgeObj = {
                                    id: "edge_" + name + "_" + j,
                                    source: "var-" + name,
                                    target: opName,
                                    label: ""
                                };

                                sdGraphEdges.push({data: edgeObj, classes: "opoutputedge"});
                            }
                        }

                        //Add variable control dependencies:
                        var vcdCount = v.controlDepsLength();
                        if (vcdCount > 0) {
                            for (var j = 0; j < vcdCount; j++) {
                                var vcd = v.controlDeps(j);

                                //2 possibilities: variable is a variable/constant/placeholder: source is from variable node
                                //Or variable is output of an op: source is from an op node
                                var vcdVariable = sdGraphVariableMap.get(vcd);
                                var sourceName;
                                var edgeLabel;
                                if (vcdVariable.type() === nd4j.graph.VarType.ARRAY) {
                                    //Control dependency: array -> variable/const/placeholder
                                    sourceName = vcdVariable.outputOfOp();
                                    sourceName = idEscapeSlashes(sourceName);
                                    edgeLabel = vcd;    //Don't need to report datatype here, data is not actually used
                                } else {
                                    //Control dependency: variable/const/placeholder -> variable/const/placeholder
                                    sourceName = "var-" + vcd;
                                    edgeLabel = "";
                                }

                                var edgeObj = {
                                    source: sourceName,
                                    target: "var-" + name,
                                    label: edgeLabel
                                };
                                sdGraphEdges.push({data: edgeObj, classes: "controldepedge"});
                            }
                        }
                    }

                    count += 1;
                }

                //Op nodes
                var mapOpNameInteger = new Map();
                sdGraphOpsMap = new Map();
                count = 0;
                opCount = sdGraphOpsList.length;
                for (var i = 0; i < sdGraphOpsList.length; i++) {
                    var o = sdGraphOpsList[i];
                    var name = o.name();
                    var opName = o.opName();
                    mapOpNameInteger.set(count, name);
                    sdGraphOpsMap.set(name, o);

                    var label = "\"" + name + "\"\n(" + opName + ")";
                    var e = o.uiLabelExtra();
                    if (e != null && e !== "") {
                        label = label + "\n" + e;
                    }

                    var opclasses = "uiop";
                    if (opName === "enter") {
                        opclasses = opclasses + " openter";
                    } else if (opName === "exit") {
                        opclasses = opclasses + " opexit";
                    } else if (opName === "next_iteration") {
                        opclasses = opclasses + " opnextiter";
                    } else if (opName === "switch") {
                        opclasses = opclasses + " opswitch";
                    } else if (opName === "merge") {
                        opclasses = opclasses + " opmerge";
                    }

                    var id = idEscapeSlashes(name);

                    var nodeObj = {
                        label: label,
                        id: id
                    };
                    sdGraphNodes.push({data: nodeObj, classes: opclasses});


                    //Add edges between ops:
                    var ol = o.outputsLength();
                    if (ol > 0) {
                        for (var j = 0; j < ol; j++) {
                            var outVarName = o.outputs(j);
                            var outVar = sdGraphVariableMap.get(outVarName);
                            var outVarInputCount = outVar.inputsForOpLength();

                            //Op -> outVar -> otherOp exists
                            //But we'll represent this as one edge in the graph only

                            var dt = dataTypeToString(outVar.datatype());

                            if (outVarInputCount > 0) {
                                for (var k = 0; k < outVarInputCount; k++) {
                                    var opName = outVar.inputsForOp(k);
                                    opName = idEscapeSlashes(opName);
                                    var edgeObj = {
                                        source: id,
                                        target: opName,
                                        label: outVarName + " (" + dt + ")"
                                    };
                                    sdGraphEdges.push({data: edgeObj, classes: "opoutputedge"});
                                }
                            }
                        }
                    }

                    //Add control dependencies:
                    var cdLength = o.controlDepsLength();
                    if (cdLength > 0) {
                        for (var j = 0; j < cdLength; j++) {
                            var varName = o.controlDeps(j);
                            //If placeholder, variable or constant, make edge from variable node
                            //If array, make edge from op node

                            var variable = sdGraphVariableMap.get(varName);
                            var dt = dataTypeToString(variable.datatype());
                            var vType = variable.type();
                            var edgeObj;
                            if (vType === nd4j.graph.VarType.CONSTANT || vType === nd4j.graph.VarType.PLACEHOLDER || vType === nd4j.graph.VarType.VARIABLE) {
                                edgeObj = {
                                    source: "var-" + varName,
                                    target: id,
                                    label: ""
                                }
                            } else {
                                var inOpName = variable.outputOfOp();
                                inOpName = idEscapeSlashes(inOpName);
                                edgeObj = {
                                    source: inOpName,
                                    target: id,
                                    label: "CD: " + varName + " (" + dt + ")"
                                };
                            }
                            sdGraphEdges.push({data: edgeObj, classes: "controldepedge"});
                        }
                    }

                    count += 1;
                }

                //Also add the outputs:
                for (var i = 0; i < sdGraphOutputs.length; i++) {
                    var outName = sdGraphOutputs[i];
                    var v = sdGraphVariableMap.get(outName);
                    var opName = v.outputOfOp();
                    opName = idEscapeSlashes(opName);
                    if (opName != null) {
                        var dt = dataTypeToString(v.datatype());
                        var shape = varShapeToString(v);
                        var n = "Output: \"" + outName + "\"\n" + varTypeToString(vType) + "\n" + dt + " " + shape;

                        var nodeObj = {
                            label: n,
                            id: "out-" + name
                        };

                        sdGraphNodes.push({data: nodeObj, classes: "uivariable output"});

                        //Also add edge:
                        var edgeObj = {
                            label: "",
                            source: opName,
                            target: "out-" + name
                        };
                        sdGraphEdges.push({data: edgeObj, classes: "opoutputedge"});
                    }
                }


                //Render the side bar:
                document.getElementById('selectedfile').innerHTML =
                    "<br><br><strong>File:</strong> " + file.name + "<br>" +
                    "<strong>Inputs:</strong> \"" + sdGraphInputs.join("\", \"") + "\"<br>" +
                    "<strong>Outputs:</strong> \"" + sdGraphOutputs.join("\", \"") + "\"<br>" +
                    "<strong>Placeholder Count:</strong> " + phCount + "<br>" +
                    "<strong>Variable Count:</strong> " + varCount + "<br>" +
                    "<strong>Constant Count:</strong> " + constCount + "<br>" +
                    "<strong>Op Count:</strong> " + opCount + "<br>";


            } else if (decoded[0] === "systeminfo") {

            } else if (decoded[0] === "startevents") {

            }

            renderContent();
        };
    }
}

function renderContent(){
    document.getElementById("samediffcontent").innerHTML = "";

    switch(selectedPage){
        case "graph":
            renderSameDiffGraph();
            break;
        case "plots":
            readAndRenderPlotsData();
            break;
        case "evaluation":
        case "performance":
            //TODO
            renderPageNotImplemented();
        default:
            renderPageNotImplemented();
    }
}




function renderPageNotImplemented(){
    document.getElementById("samediffcontent").innerHTML = "<br><br>Page not yet implemented: " + selectedPage + "<br>";
}

samediffgraphlayout = "klay";
klaylayout = "DOWN";
function setLayout(newLayout){
    //spread( cytoscape );
    if(newLayout === "klay_down"){
        klaylayout = "DOWN";
        newLayout = "klay";
    } else if(newLayout === "klay_lr"){
        klaylayout = "RIGHT";
        newLayout = "klay";
    }
    samediffgraphlayout = newLayout;
    renderContent();
}

function idEscapeSlashes(input){
    return input.replace(new RegExp('/', 'g'), '__');
}

function idRestoreSlashes(input){
    return input.replace(new RegExp('__', 'g'), '/');
}