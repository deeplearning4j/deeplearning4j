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

function fileSelect(evt){
    var output = [];
    file = evt.target.files[0];
    output.push('<li><strong>', escape(file.name), '</strong> (', file.type || 'n/a', ') - ',
        file.size, ' bytes, last modified: ',
        file.lastModifiedDate ? file.lastModifiedDate.toLocaleDateString() : 'n/a',
        '</li>');
    document.getElementById('list').innerHTML = '<ul>' + output.join('') + '</ul>';

    console.log("About to render graph: file " + file.name);
    renderSameDiffGraph();
}



function renderSameDiffGraph(){

    var nodeNames = ["a", "b"];
    var edges = [["a","b"]];

    //First: load data
    if(file) {
        var fr = new FileReader();
        var fileData = new Blob([file]);
        fr.readAsArrayBuffer(fileData);
        fr.onload = function(){
            var arrayBuffer = fr.result
            var bytes = new Uint8Array(arrayBuffer);
            console.log(bytes);

            var lengths = extractHeaders(bytes);
            var headerLength = lengths[0];
            var contentLength = lengths[1];

            console.log("Header/content lengths: " + headerLength + ", " + contentLength);

            //https://gist.github.com/alexvictoor/0b76764857b780ad8f83

            var headerSlice = bytes.slice(8, 8+headerLength);
            var buffer = new flatbuffers.ByteBuffer(headerSlice);

            var contentSlice = bytes.slice(8+headerLength, 8+headerLength+contentLength);
            var bufferContent = new flatbuffers.ByteBuffer(contentSlice);
            var graphStructure = nd4j.graph.UIGraphStructure.getRootAsUIGraphStructure(bufferContent);
            var inLength = graphStructure.inputsLength();
            var inputs = [];
            for( var i=0; i<inLength; i++ ){
                var inName = graphStructure.inputs(i);
                inputs.push(inName);
            }
            console.log("Inputs: " + inputs);

            var outLength = graphStructure.outputsLength();
            var outputs = [];
            for( var i=0; i<outLength; i++ ){
                var outName = graphStructure.outputs(i);
                outputs.push(outName);
            }
            console.log("Outputs: " + outputs);
        };
        // console.log("Number of bytes: " + bytes.length());
    }

    var nodes = [];
    var edges = [];

    for(var i=0; i<4; i++ ){
        var obj = {
            id: i,
            name: "vertex-" + i,
            faveColor: "#4A235A",
            faveShape: "triangle"
        };
        nodes.push({ data: obj} );

        if( i > 0 ){
            var e = {
                source: i-1,
                target: i,
                faveColor: '#A9A9A9',
                strength: 100
            };
            edges.push({ data: e} );
        }
    }

    var cy = cytoscape({

        container: document.getElementById('graphdiv'), // container to render in

        layout: {
            name: 'dagre',
            padding: 10
        },

        elements: {
            nodes: nodes,
            edges: edges
        },

        style: [ // the stylesheet for the graph
            {
                selector: 'node',
                style: {
                    'background-color': '#666',
                    'label': 'data(id)'
                }
            },
            {
                selector: "edge",
                style: {
                    'curve-style': 'bezier',        //Necessary for arrow rendering, default style doesn't have them
                    'width': 2,
                    'line-color': '#ccc',
                    'target-arrow-shape': 'triangle',
                    'source-arrow-shape': 'none',
                    'source-arrow-color': '#000',
                    'target-arrow-color': '#000'
                }
            }
        ]
    });
}