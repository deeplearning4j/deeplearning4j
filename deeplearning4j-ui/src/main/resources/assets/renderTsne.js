  /*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

var x = [];
  var y = [];
  var name3 = [];




var tx=0, ty=0;
var ss=1;

var svg;
var xMax = 0, xMin = 0, yMax = 0, yMin = 0;

var fx;
var fy;

function zoomHandler() {
  tx = d3.event.translate[0];
  ty = d3.event.translate[1];
  ss = d3.event.scale;
  console.log('zoom called. Scale: ' + ss + " tX: " + tx + " tY: " + ty);
  svg.selectAll('circle')
      .data(name3)
      .attr("transform", function(d, i) { return "translate(" +
                                            ((x[i]*20*ss + tx) + 400) + "," +
                                            ((y[i]*20*ss + ty) + 400) + ")";
                                            });
  svg.selectAll('text')
        .data(name3)
        .attr("transform", function(d, i) { return "translate(" +
                                              ((x[i]*20*ss + tx) + 400) + "," +
                                              ((y[i]*20*ss + ty) + 400) + ")";
                                              });
}



function drawEmbedding() {
    $("#embed").empty();
    var div = d3.select("#embed");

    fx = d3.scale.linear()
            .domain([xMin, xMax])
            .range([0, 1024])

    fy = d3.scale.linear()
            .domain([yMin, yMax])
            .range([600, 0]);



    svg = div.append("svg") // svg is global
    .attr("width", "1024")
    .attr("height", "700");


    svg.selectAll("circle")
        .data(name3)
        .enter()
        .append("circle")
/*        .attr("transform", function(d, i) { return "translate(" +
                                                  ((x[i]*20*ss + tx) + 400) + "," +
                                                  ((y[i]*20*ss + ty) + 400) + ")"; });
*/


        .attr("cx", function(d, i) {
        			   		return fx(x[i]);
        			   })
        			   .attr("cy", function(d, i) {
        			   		return fy(y[i]);
        			   })
        			   .attr("r", 2);

    svg.selectAll("text")
    			   .data(name3)
    			   .enter()
    			   .append("text")
    			   .text(function(d,i) {
    			   		return name3[i];
    			   })
    			   .attr("x", function(d,i) {
    			   		return fx(x[i]);
    			   })
    			   .attr("y", function(d, i) {
    			   		return fy(y[i]);
    			   })
    			   .attr("font-family", "sans-serif")
    			   .attr("font-size", "11px")
    			   .attr("fill", "red");
/*
    var g = svg.selectAll(".b")
      .data(name3)
      .enter().append("g")
      .attr("class", "u");


    g.append("text")
      .attr("text-anchor", "top")
      .attr("font-size", 12)
      .attr("fill", "#333")
      .text(function(d) { return d; });

  */

/*
    svg.selectAll('.u')
    .data(name3)
    .attr("transform", function(d, i) { return "translate(" +
                                          ((x[i]*20*ss + tx) + 400) + "," +
                                          ((y[i]*20*ss + ty) + 400) + ")"; });
                                          */

    var zoomListener = d3.behavior.zoom()
      .x(fx)
      .y(fy)
      .scaleExtent([0.000001, 1000])
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
        x = new Array();
        y = new Array();
        name3 = new Array();
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

      /*
        This values could be pushed for debug purposes

        x.push(10);
        y.push(15);
        name3.push("alpha");
      */

      xMax = d3.max(x);
      xMin = d3.min(x);

      yMax = d3.max(y);
      yMin = d3.min(y);


      drawEmbedding();
    });
}



$(window).load(drawTsne());
