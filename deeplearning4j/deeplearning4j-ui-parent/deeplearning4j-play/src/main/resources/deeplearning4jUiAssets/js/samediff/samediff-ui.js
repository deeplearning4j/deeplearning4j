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

function toggleSidebar(){
    $('#samediffsidebar').toggleClass('sidebarhidden');
}

function samediffSetPage(pageName){
    console.log("Selected page: " + pageName);
}


nodes = [];
edges = [];

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

    //First: load data
    if (file) {
        var fr = new FileReader();
        var fileData = new Blob([file]);
        fr.readAsArrayBuffer(fileData);
        fr.onload = function () {
            var arrayBuffer = fr.result
            var bytes = new Uint8Array(arrayBuffer);
            console.log(bytes);

            var lengths = extractHeaders(bytes);
            var headerLength = lengths[0];
            var contentLength = lengths[1];

            console.log("Header/content lengths: " + headerLength + ", " + contentLength);

            //https://gist.github.com/alexvictoor/0b76764857b780ad8f83

            var headerSlice = bytes.slice(8, 8 + headerLength);
            var contentSlice = bytes.slice(8 + headerLength, 8 + headerLength + contentLength);

            var decoded = decodeStaticInfo(headerSlice, contentSlice);

            console.log("Decoded header message: " + decoded[0]);

            if (decoded[0] === "graph") {
                var opCount = 0;
                var phCount = 0;
                var varCount = 0;
                var constCount = 0;

                nodes.length = 0;
                edges.length = 0;
                var graph = decoded[1];
                var inputs = uiGraphGetInputs(graph);
                var outputs = uiGraphGetOutputs(graph);
                var variables = uiGraphGetVariables(graph);
                var ops = uiGraphGetOps(graph);
                console.log("Inputs: " + inputs);
                console.log("Outputs: " + outputs);
                console.log("Variables: " + variables);
                console.log("Ops: " + ops);

                var mapVarNameInteger = new Map();
                var mapVars = new Map();            //Key: variable name, value: variable
                var count = 0;
                for (var i = 0; i < variables.length; i++) {
                    var v = variables[i];
                    var name = v.name();
                    mapVarNameInteger.set(count, name);
                    mapVars.set(name, v);
                }

                for (var i = 0; i < variables.length; i++) {
                    var v = variables[i];
                    var name = v.name();
                    //Add variables/constants/placeholders as a node
                    var vType = v.type();

                    switch(vType){
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

                    if(vType === nd4j.graph.VarType.CONSTANT || vType === nd4j.graph.VarType.PLACEHOLDER || vType === nd4j.graph.VarType.VARIABLE ){
                        var dt = dataTypeToString(v.datatype());
                        var shape = varShapeToString(v);
                        var n = "\"" + name + "\"\n" + varTypeToString(vType) + "\n" + dt + " " + shape;

                        var extraLabel = v.uiLabelExtra();
                        if(extraLabel != null && extraLabel !== ""){
                            n = n + "\n" + extraLabel;
                        }


                        if(vType === nd4j.graph.VarType.CONSTANT){
                            var constArr = v.constantValue();
                            if(constArr != null){
                                if(constArr.shapeLength() === 0 && constArr.bufferLength() > 0){
                                    var scalar = scalarFromFlatArray(constArr);
                                    if(scalar != null && scalar !== ""){
                                        n = n + "\nScalar val: " + scalar;
                                    }
                                }
                            }
                        }


                        var nodeObj = {
                            label: n,
                            id: "var-" + name,
                            name: "var-" +name
                        };

                        var renderStyle = "";
                        if(vType === nd4j.graph.VarType.VARIABLE){
                            renderStyle = "uivariable variable";
                        } else if(vType === nd4j.graph.VarType.PLACEHOLDER){
                            renderStyle = "uivariable placeholder";
                        } else if(vType === nd4j.graph.VarType.CONSTANT){
                            renderStyle = "uivariable constant";
                        }

                        nodes.push({data: nodeObj, classes: renderStyle});

                        if(v.inputsForOpLength() > 0){
                            for(var j=0; j<v.inputsForOpLength(); j++ ){
                                var opName = v.inputsForOp(j);
                                var edgeObj = {
                                    id: "edge_" + name + "_" + j,
                                    source: "var-" + name,
                                    target: opName,
                                    label: ""
                                };

                                edges.push({data: edgeObj, classes:"opoutputedge"});
                            }
                        }

                        //Add variable control dependencies:
                        var vcdCount = v.controlDepsLength();
                        if(vcdCount > 0){
                            for(var j=0; j<vcdCount; j++ ){
                                var vcd = v.controlDeps(j);

                                //2 possibilities: variable is a variable/constant/placeholder: source is from variable node
                                //Or variable is output of an op: source is from an op node
                                var vcdVariable = mapVars.get(vcd);
                                var sourceName;
                                var edgeLabel;
                                if(vcdVariable.type() === nd4j.graph.VarType.ARRAY){
                                    //Control dependency: array -> variable/const/placeholder
                                    sourceName = vcdVariable.outputOfOp();
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
                                edges.push({data: edgeObj, classes:"controldepedge"});
                            }
                        }
                    }

                    count += 1;
                }

                //Op nodes
                var mapOpNameInteger = new Map();
                var mapOp = new Map();
                count = 0;
                opCount = ops.length;
                for (var i = 0; i < ops.length; i++) {
                    var o = ops[i];
                    var name = o.name();
                    var opName = o.opName();
                    mapOpNameInteger.set(count, name);
                    mapOp.set(name, o);

                    var label = "\"" + name + "\"\n(" + opName + ")";
                    var e = o.uiLabelExtra();
                    if(e != null && e !== ""){
                        label = label + "\n" + e;
                    }

                    var opclasses = "uiop";
                    if(opName === "enter"){
                        opclasses = opclasses + " openter";
                    } else if(opName === "exit"){
                        opclasses = opclasses + " opexit";
                    } else if(opName === "next_iteration"){
                        opclasses = opclasses + " opnextiter";
                    } else if(opName === "switch"){
                        opclasses = opclasses + " opswitch";
                    } else if(opName === "merge"){
                        opclasses = opclasses + " opmerge";
                    }

                    var nodeObj = {
                        label: label,
                        id: name,
                        name: name
                    };
                    nodes.push({data: nodeObj, classes:opclasses});


                    //Add edges between ops:
                    var ol = o.outputsLength();
                    if(ol > 0){
                        for( var j =0; j<ol; j++ ){
                            var outVarName = o.outputs(j);
                            var outVar = mapVars.get(outVarName);
                            var outVarInputCount = outVar.inputsForOpLength();

                            //Op -> outVar -> otherOp exists
                            //But we'll represent this as one edge in the graph only

                            var dt = dataTypeToString(outVar.datatype());

                            if(outVarInputCount > 0){
                                for( var k=0; k<outVarInputCount; k++ ){
                                    var opName = outVar.inputsForOp(k);
                                    var edgeObj = {
                                        source: name,
                                        target: opName,
                                        label: outVarName + " (" + dt + ")"
                                    };
                                    edges.push({data: edgeObj, classes:"opoutputedge"});
                                }
                            }
                        }
                    }

                    //Add control dependencies:
                    var cdLength = o.controlDepsLength();
                    if(cdLength > 0){
                        for( var j=0; j<cdLength; j++ ) {
                            var varName = o.controlDeps(j);
                            //If placeholder, variable or constant, make edge from variable node
                            //If array, make edge from op node

                            var variable = mapVars.get(varName);
                            var dt = dataTypeToString(variable.datatype());
                            var vType = variable.type();
                            var edgeObj;
                            if(vType === nd4j.graph.VarType.CONSTANT || vType === nd4j.graph.VarType.PLACEHOLDER || vType === nd4j.graph.VarType.VARIABLE ){
                                edgeObj = {
                                    source: "var-" + varName,
                                    target: opName,
                                    label: ""
                                }
                            } else {
                                var inOpName = variable.outputOfOp();
                                edgeObj = {
                                    source: inOpName,
                                    target: opName,
                                    label: "CD: " + varName + " (" + dt + ")"
                                };
                            }
                            edges.push({data: edgeObj, classes:"controldepedge"});
                        }
                    }

                    count += 1;
                }

                //Also add the outputs:
                for( var i=0; i<outputs.length; i++ ){
                    var outName = outputs[i];
                    var v = mapVars.get(outName);
                    var opName = v.outputOfOp();
                    if(opName != null){
                        var dt = dataTypeToString(v.datatype());
                        var shape = varShapeToString(v);
                        var n = "Output: \"" + outName + "\"\n" + varTypeToString(vType) + "\n" + dt + " " + shape;

                        var nodeObj = {
                            label: n,
                            id: "out-" + name,
                            name: "out-" +name
                        };

                        nodes.push({data: nodeObj, classes:"uivariable output"});

                        //Also add edge:
                        var edgeObj = {
                            label: "",
                            source: opName,
                            target: "out-" + name
                        };
                        edges.push({data: edgeObj, classes:"opoutputedge"});
                    }
                }


                //Render the side bar:
                document.getElementById('sidebartop').innerHTML =
                    "<br><br><strong>File:</strong> " + file.name + "<br>" +
                    "<strong>Inputs:</strong> \"" + inputs.join("\", \"") + "\"<br>" +
                    "<strong>Outputs:</strong> \"" + outputs.join("\", \"") + "\"<br>" +
                    "<strong>Placeholder Count:</strong> " + phCount + "<br>" +
                    "<strong>Variable Count:</strong> " + varCount + "<br>" +
                    "<strong>Constant Count:</strong> " + constCount + "<br>" +
                    "<strong>Op Count:</strong> " + opCount + "<br>";


            } else if (decoded[0] === "systeminfo") {

            } else if (decoded[0] === "startevents") {

            }

            renderSameDiffGraph();
        };
        // console.log("Number of bytes: " + bytes.length());
    }
}


function renderSameDiffGraph() {

    if (nodes) {

        var cy = cytoscape({

            container: document.getElementById('graphdiv'), // container to render in

            layout: {
                name: samediffgraphlayout,
                padding: 10,
                klay : {
                    direction: klaylayout
                }
            },

            elements: {
                nodes: nodes,
                edges: edges
            },
            style: fetch('/assets/js/samediff/cytoscape-style.json').then(function(res){
                return res.json();
            }),
            wheelSensitivity: 0.2
        });
    }
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
    renderSameDiffGraph();
}