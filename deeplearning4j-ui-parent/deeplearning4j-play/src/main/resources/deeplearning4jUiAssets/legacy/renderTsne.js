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

var height=700;
var width=1024;


var x = [];
  var y = [];
  var name3 = [];




var tx=0, ty=0;
var ss=1;

var svg;
var xMax = 0, xMin = 0, yMax = 0, yMin = 0;

var fx;
var fy;

var xAxis;
var yAxis;

function zoomHandler() {

  svg.select(".x.axis").call(xAxis);
  svg.select(".y.axis").call(yAxis);

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


var marginLeft = 100;
var widthExMargin = width - marginLeft;
var marginTop = 100;
var heightExMargin = height - marginTop;

function drawEmbedding() {
    $("#embed").empty();
    var div = d3.select("#embed");

    xScale = d3.scale.linear().range([0,widthExMargin]);
    yScale = d3.scale.linear().range([heightExMargin, 0]);

    xAxis = d3.svg.axis().scale(xScale)
        .innerTickSize(-height) //Used as grid line
        .orient("bottom").ticks(5);

    yAxis = d3.svg.axis().scale(yScale)
        .innerTickSize(-width)  //Used as grid line
        .orient("left").ticks(5);

    svg = div.append("svg") // svg is global
        .attr("width", widthExMargin)
        .attr("height", heightExMargin)
        .attr("transform", "translate(" + marginLeft + "," + marginTop + ")");

    // Add the X Axis
    xAxisNode = svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + heightExMargin + ")")
        .style("stroke", "#000")
        .style("stroke-width", "1.0")
        .style("fill", "#000")
        .call(xAxis);
    xAxisNode.selectAll('text').style("stroke-width",0).style("fill","#000000");


    // Add the Y Axis
    var yAxisNode = svg.append("g")
        .attr("class", "y axis")
        .style("stroke", "#000")
        .style("stroke-width", "1.0")
        .style("fill", "#000")
        .call(yAxis);
    yAxisNode.selectAll('text').style("stroke-width",0).style("fill","#000000");


    xMin = d3.min(x);
    xMax = d3.max(x);
    yMin = d3.min(y);
    yMax = d3.max(y);

    xScale.domain([xMin, xMax]);
    yScale.domain([yMin, yMax]);

    //Add the data
    var data = x.map(function (d, i) {
        return {'xPos': x[i], 'yPos': y[i]};
    });

    svg.selectAll("circle")
        .data(data)
        .enter()
        .append("circle")
        .style("fill", "#000000")
        .attr("r", "3")
        .attr("cx", function (d) {
            return xScale(d['xPos']);
        })
        .attr("cy", function (d) {
            return yScale(d['yPos']);
        });

    //
    // fx = d3.scale.linear()
    //     .domain([xMin, xMax])
    //     .range([0, width])
    //
    // fy = d3.scale.linear()
    //     .domain([yMin, yMax])
    //     .range([height, 0]);
    //
    // console.log("xMin/xMax: " + xMin + "," + xMax + "; yMin/yMax: " + yMin + "," + yMax);
    // console.log("width = " + width + ", height = " + height);
    //
    // //Define X axis
    // xAxis = d3.svg.axis()
		// .scale(fx)
		// .orient("bottom")
		// .tickSize(-height)
    //     .ticks(10)
		// .tickFormat(d3.format("s"));
    //
    // //Define Y axis
    // yAxis = d3.svg.axis()
		// .scale(fy)
		// .orient("left")
		// .ticks(5)
		// .tickSize(-width)
		// .tickFormat(d3.format("s"));
    //
    //
    // svg = div.append("svg") // svg is global
    // .attr("width", width)
    // .attr("height", height);
    //
    //
    // svg.append("g")
		// .attr("class", "x axis")
		// .attr("transform", "translate(0," + height + ")")
		// .call(xAxis);
    //
    // svg.append("g")
		// .attr("class", "y axis")
		// .call(yAxis);
    //
    //
    //
    // svg.selectAll("circle")
    //     .data(name3)
    //     .enter()
    //     .append("circle")
    //     .attr("cx", function(d, i) {
    //     			   		return fx(x[i]);
    //     			   })
    //     			   .attr("cy", function(d, i) {
    //     			   		return fy(y[i]);
    //     			   })
    //     			   .attr("r", 2);
    //
    // svg.selectAll("text")
    //    .data(name3)
    //    .enter()
    //    .append("text")
    //    .text(function(d,i) {
    // 			   		return name3[i];
    // 			   })
    //    .attr("x", function(d,i) {
    // 			   		return fx(x[i]);
    // 			   })
    //    .attr("y", function(d, i) {
    // 			   		return fy(y[i]);
    // 			   })
    //    .attr("font-family", "sans-serif")
    //    .attr("font-size", "11px")
    //    .attr("fill", "red");
    //
    //
    // var zoomListener = d3.behavior.zoom()
    //     .x(fx)
    //     .y(fy)
    //     .scaleExtent([0.00000001, 100000])
    //     .center([0,0])
    //     .on("zoom", zoomHandler);
    // zoomListener(svg);

}


function drawTsne() {
    var sid = getParameterByName("sid");
    if (sid == undefined) sid = "UploadedFile";


 $.ajax({
     url: "/tsne/coords/" + sid,
    cache: false
  })
    .done(function( data ) {
        x = [];
        y = [];
        name3 = [];
        if (data.length > 0 ) {
          for(var i = 0; i < data.length; i++) {
             var split = data[i].split(',');
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

        // console.log("xMin: " + xMin);
        // console.log("xMax: " + xMax);
        //
        // console.log("yMin: " + yMin);
        // console.log("yMax: " + yMax);

        drawEmbedding();
      } else {
        $.notify({
        	title: '<strong>No t-SNE data available!</strong>',
        	message: 'Please upload something, or check again later.'
        },{
        	type: 'danger',
        	placement: {
            		from: "top",
            		align: "center"
            	},
        });
      }
    });
}



$(window).load(drawTsne());
