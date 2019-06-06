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

var x = [];
var y = [];
var name3 = [];




var tx = 0, ty = 0;
var ss = 1;
function zoomHandler() {
    tx = d3.event.translate[0];
    ty = d3.event.translate[1];
    ss = d3.event.scale;
    console.log('zoom called');
    svg.selectAll('.u')
        .data(name3)
        .attr("transform", function(d, i) { return "translate(" +
            ((x[i]*20*ss + tx) + 400) + "," +
            ((y[i]*20*ss + ty) + 400) + ")";
        });
}

var svg;

function drawEmbedding() {
    $("#embed").empty();
    var div = d3.select("#embed");


    svg = div.append("svg") // svg is global
        .attr("width", 60000)
        .attr("height", 60000);

    var g = svg.selectAll(".b")
        .data(name3)
        .enter().append("g")
        .attr("class", "u");


    g.append("text")
        .attr("text-anchor", "top")
        .attr("font-size", 12)
        .attr("fill", "#333")
        .text(function(d) { return d; });


    svg.selectAll('.u')
        .data(name3)
        .attr("transform", function(d, i) { return "translate(" +
            ((x[i]*20*ss + tx) + 400) + "," +
            ((y[i]*20*ss + ty) + 400) + ")"; });

    var zoomListener = d3.behavior.zoom()
        .scaleExtent([0.1, 10])
        .center([0,0])
        .on("zoom", zoomHandler);
    zoomListener(svg);
}


function drawTsne() {
    $.ajax({
        url: "/api/coords",
        cache: false
    })
        .done(function( data ) {
            for(var i = 0; i < data.length; i++) {
                var split = data[i].split(',');
                console.log(split.length);
                var xCoord = split[0];
                var yCoord = split[1];
                var name2 = split[2];
                x.push(xCoord);
                y.push(yCoord);
                name3.push(name2);
            }

            drawEmbedding();
        });

}



$(document).ready(function() {

});

