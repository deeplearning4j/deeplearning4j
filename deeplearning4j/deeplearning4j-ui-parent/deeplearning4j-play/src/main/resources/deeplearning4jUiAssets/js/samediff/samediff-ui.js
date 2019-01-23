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

    //First: load data
    // var file =

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