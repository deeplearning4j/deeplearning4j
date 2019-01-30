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




function renderSameDiffGraph() {
    document.getElementById("samediffcontent").innerHTML = "<div id=\"graphdiv\" style=\"height: 100%; width: 100%; display: table\"></div>";

    if (sdGraphNodes) {

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



function onGraphNodeClick(nodeId){
    var element = $("#sidebarmid");

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
        // var inVarsStr = "";
        for( var i=0; i<op.inputsLength(); i++ ){
            var inName = op.inputs(i);
            var inVar = sdGraphVariableMap.get(inName);     //nd4j.graph.UIVariable
            var dtype = dataTypeToString(inVar.datatype());
            var shape = varShapeToString(inVar);
            inVars.push(inName + " (" + dtype + (shape == null || shape === "" ? "" : "," + shape ) + ")");
            // inVarsStr = inVarsStr + op.inputs(i) + "<br>";
        }

        var outVars = [];
        // var outVarsStr = "";
        var inputsForOps = new Set();
        for( var i=0; i<op.outputsLength(); i++ ){
            var outName = op.outputs(i);
            var outVar = sdGraphVariableMap.get(outName);     //nd4j.graph.UIVariable
            var dtype = dataTypeToString(outVar.datatype());
            var shape = varShapeToString(outVar);
            outVars.push(outName + " (" + dtype + (shape == null || shape === "" ? "" : "," + shape) + ")");
            // outVarsStr = outVarsStr + op.inputs(i) + "<br>";
            var inputsForLength = outVar.inputsForOpLength();
            for( var j=0; j<inputsForLength; j++ ){
                inputsForOps.add(outVar.inputsForOp(j));
            }
        }

        extra = "<b>Op Name:</b> " + op.opName() + "<br>" +
        "<b>Input Variables:</b><br>" +
        inVars.join("<br>") + "<br>" +
        "<b>Output Variables:</b> " + "" + "<br>" +
        outVars.join("<br>") + "<br>" +
        "<b>Inputs For Ops:</b><br>" +
            (inputsForOps.length === 0 ? "<none>" : Array.from(inputsForOps).join("<br>")) + "<br>";


    }

    document.getElementById("sidebarmid").innerHTML = "<br><br>\n" +
        "<b>Selected Node:</b><br>" +
        "<b>Name:</b> " + name + "<br>" +
        "<b>Type:</b> " + type + "<br>" +
        extra;
}

function onGraphNodeSearch(){
    var value = document.getElementById("findnodetxt").value;
    console.log("SEARCH TERM: " + value);

    var results = [];
    if(value != null && value !== ""){
        //Iterate over ops, find ones that contain the search string
        var keySet = sdGraphOpsMap.keys();
        var values = sdGraphOpsMap.values();    //Iterator<nd4j.graph.UIOp>
        // for( var v in values ){
        // while(values.hasNe)
        for(var i=0; i<sdGraphOpsList.length; i++ ){
            var op = sdGraphOpsList[i];
            var name = op.name();
            if(name.includes(value)){
                results.push(name);
            }
        }
    }

    var listHtml = "<ul>\n";
    for( var i=0; i<results.length; i++ ){
        listHtml = listHtml + "<li onclick='onGraphNodeSearchResultClick(\"" + results[i] + "\")'>" + results[i] + "</li>\n";
    }
    listHtml = listHtml + "</ul>";
    document.getElementById("findnoderesults").innerHTML = listHtml;
}

function onGraphNodeSearchResultClick( clicked ){
    console.log("CLICKED ON SEARCH RESULT: " + clicked);
}