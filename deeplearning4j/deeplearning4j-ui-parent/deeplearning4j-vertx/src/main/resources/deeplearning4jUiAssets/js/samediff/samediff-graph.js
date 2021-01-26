/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
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




function renderSameDiffGraph() {
    document.getElementById("samediffcontent").innerHTML = "<div id=\"graphdiv\" style=\"height: 100%; width: 100%; display: table\"></div>";

    if (sdGraphNodes) {

        cy = cytoscape({

            container: document.getElementById('graphdiv'), // container to render in

            layout: {
                name: samediffgraphlayout,
                padding: 10,
                klay : {
                    direction: klaylayout
                }
            },

            elements: {
                nodes: sdGraphNodes,
                edges: sdGraphEdges
            },
            style: fetch('/assets/js/samediff/cytoscape-style.json').then(function(res){
                return res.json();
            }),
            wheelSensitivity: 0.2
        });

        // cy.nodes().on("click", function(e){
        //     var clickedNode = e.target;
        //     console.log("Clicked node: " + clickedNode);
        // });
        // cy.nodes().once('click', function(e){
        //     var ele = e.target;
        //     console.log('clicked ' + ele.id());
        // });
        cy.on('click', 'node', function(e){
            var ele = e.target;
            console.log('clicked ' + ele.id());
            onGraphNodeClick(ele.id());
        });
    }
}



function onGraphNodeClick(/*String*/ node){
    var element = $("#sidebarmid");

    var nodeId = idRestoreSlashes(node);    //"while__Enter" -> "while/Enter"

    //Next, find all inputs and outputs...
    var type = "-";
    var extra = "";
    var name = "";
    if(nodeId.startsWith("var-")){
        name = nodeId.substring(4);
        if(sdGraphVariableMap.has(name)){
            type = "Variable";
            var v = sdGraphVariableMap.get(name);
            type = varTypeToString(v.type());
            var dtype = dataTypeToString(v.datatype());
            var shape = varShapeToString(v);
            extra = "<b>Data type:</b> " + dtype + "<br><b>Shape:</b> " + shape;
        }
    } else {
        type = "Op";
        name = nodeId;
        var op = sdGraphOpsMap.get(name);   //Map<String,nd4j.graph.UIOp>

        var inVars = [];
        for( var i=0; i<op.inputsLength(); i++ ){
            var inName = op.inputs(i);
            var inVar = sdGraphVariableMap.get(inName);     //nd4j.graph.UIVariable
            var outputOfOpName = inVar.outputOfOp();
            var pre = "";
            var post = "";
            if(outputOfOpName != null && outputOfOpName !== ""){
                // var op = sdGraphOpsMap.get(outputOfOpName);
                pre = "<span onclick='onGraphNodeClick(\"" + outputOfOpName + "\");centerViewOnNode(\"" + outputOfOpName + "\");'>";
                post = "</span>";
            } else {
                //Not the output of an op, therefore must be a variable node
                pre = "<span onclick='onGraphNodeClick(\"var-" + inName + "\");centerViewOnNode(\"var-" + inName + "\");'>";
                post = "</span>";
            }
            var dtype = dataTypeToString(inVar.datatype());
            var shape = varShapeToString(inVar);
            inVars.push(pre + inName + " (" + dtype + (shape == null || shape === "" ? "" : "," + shape ) + ")" + post);
        }

        var outVars = [];
        // var outVarsStr = "";
        var inputsForOps = new Set();
        var len = op.outputsLength();
        for( var i=0; i<len; i++ ){
            var outName = op.outputs(i);
            var outVar = sdGraphVariableMap.get(outName);     //nd4j.graph.UIVariable
            var dtype = dataTypeToString(outVar.datatype());
            var shape = varShapeToString(outVar);
            outVars.push(outName + " (" + dtype + (shape == null || shape === "" ? "" : "," + shape) + ")");
            // outVarsStr = outVarsStr + op.inputs(i) + "<br>";
            var inputsForLength = outVar.inputsForOpLength();
            for( var j=0; j<inputsForLength; j++ ){
                var outOpName = outVar.inputsForOp(j);
                inputsForOps.add(outOpName);
            }
        }
        var outOpStr = "";
        inputsForOps = Array.from(inputsForOps);
        for(var i=0; i<inputsForOps.length; i++ ){
            var s = inputsForOps[i];
            outOpStr = outOpStr + "<span onclick='onGraphNodeClick(\"" + s + "\");centerViewOnNode(\"" + s + "\");'>" + s + "</span><br>";
        }

        extra = "<b>Op Name:</b> " + op.opName() + "<br>" +
        "<b>Input Variables:</b><br>" +
        inVars.join("<br>") + "<br>" +
        "<b>Output Variables:</b> " + "" + "<br>" +
        outVars.join("<br>") + "<br>" +
        "<b>Inputs For Ops:</b><br>" + outOpStr;


    }

    document.getElementById("sidebarmid-content").innerHTML =
        "<b>Name:</b> " + name + "<br>" +
        "<b>Type:</b> " + type + "<br>" +
        extra;
}

function onGraphNodeSearch(){
    var value = document.getElementById("findnodetxt").value;

    var results = [];
    if(value != null && value !== ""){
        // for( var v in values ){
        // while(values.hasNe)
        for(var i=0; i<sdGraphOpsList.length; i++ ){
            var op = sdGraphOpsList[i];
            var name = op.name();
            if(name.includes(value)){
                results.push(name);
            }
        }

        //Also contant/placeholder/variable variables (these are rendered as nodes in graph)
        for(var i=0; i<sdGraphVariableNames.length; i++ ){
            var n = sdGraphVariableNames[i];
            var vType = sdGraphVariableMap.get(n).type();
            if (vType === nd4j.graph.VarType.CONSTANT || vType === nd4j.graph.VarType.PLACEHOLDER || vType === nd4j.graph.VarType.VARIABLE) {
                if(n.includes(value)){
                    results.push(n);
                }
            }
        }
    }

    var listHtml = "<ul>\n";
    for( var i=0; i<results.length; i++ ){
        listHtml = listHtml + "<li onclick='centerViewOnNode(\"" + results[i] + "\")'>" + results[i] + "</li>\n";
    }
    listHtml = listHtml + "</ul>";
    document.getElementById("findnoderesults").innerHTML = listHtml;
}

function centerViewOnNode(/*String*/ clicked ){
    //Find the node, and center the view on it
    // var node = cy.$("#" + clicked);  //"The selector `#while/Enter`is invalid"
    var id = idEscapeSlashes(clicked);
    if(sdGraphVariableMap.has(id)){
        id = "var-" + id;
    }
    var node = cy.$('#' + id);
    cy.center(node);
}