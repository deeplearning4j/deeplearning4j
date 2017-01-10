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

var height = 700;
var width = 1024;


var x = [];
var y = [];
var name3 = [];

var renderSpace;
var xMax = 0, xMin = 0, yMax = 0, yMin = 0;

var xAxis;
var yAxis;

var xScale;
var yScale;

var currTsneSessionID = null;

function zoomFunction() {

    var translateX = d3.event.translate[0];
    var translateY = d3.event.translate[1];
    var currentScale = d3.event.scale;
    // console.log('zoom called. Scale: ' + currentScale + " translateX: " + translateX + " translateY: " + translateY);

    //Redraw the x and y axis:
    renderSpace.select(".x.axis").call(xAxis);
    renderSpace.select(".y.axis").call(yAxis);
    xAxisNode.selectAll('text').style("stroke-width", 0).style("fill", "#000000");
    yAxisNode.selectAll('text').style("stroke-width", 0).style("fill", "#000000");

    renderSpace.selectAll('circle')
        .attr("cx", function (d, i) {
            // return xScale(d.cx);
            return xScale(d['xPos']);
        })
        .attr("cy", function (d, i) {
            // return yScale(d.cy);
            return yScale(d['yPos']);
        });

    renderSpace.selectAll(".tsneTextLabels")
        .attr("x", function (d, i) {
            // return xScale(d.cx);
            return xScale(d['xPos']);
            // return xScale(d);
        })
        .attr("y", function (d, i) {
            // return yScale(d.cy);
            return yScale(d['yPos']);
            // return yScale(d);
        });

}

var circleRadius = 2.0;
var textSize = "11pt";
var textColor = "red";
var gridColor = "#dddddd";

var marginLeft = 60;
var marginTop = 10;
var marginBottom = 30;
var marginRight = 10;
var widthExMargin = width - marginLeft - marginRight;
var heightExMargin = height - marginTop - marginBottom;

var zoom;


function centerFn(){
    return d3.zoomIdentity.scale(0.00001);
}

function drawEmbedding() {
    $("#embed").empty();
    var div = d3.select("#embed");

    xScale = d3.scale.linear().range([0, widthExMargin]);
    yScale = d3.scale.linear().range([heightExMargin, 0]);

    xAxis = d3.svg.axis().scale(xScale)
        .innerTickSize(-heightExMargin) //Used as grid line
        .orient("bottom").ticks(5);

    yAxis = d3.svg.axis().scale(yScale)
        .innerTickSize(-widthExMargin)  //Used as grid line
        .orient("left").ticks(5);

    svg = div.append("svg") // svg is global
        .attr("width", width)
        .attr("height", height);

    var xMin = d3.min(x);
    var xMax = d3.max(x);
    var yMin = d3.min(y);
    var yMax = d3.max(y);

    // zoom = d3.behavior.zoom()
    var xRange = xMax - xMin;
    var yRange = yMax - yMin;
    var initialScale = 1.0 / Math.max(xRange, yRange);
    zoom = d3.behavior.zoom()
        .x(xScale)
        .y(yScale)
        .translate([widthExMargin/2, heightExMargin/2])
        .scale(initialScale)
        .on("zoom", zoomFunction);

    var initialXScale;
    var initialYScale;
    if(xRange > yRange){
        initialXScale = 1.0;
        initialYScale = xRange / yRange;
    } else {
        initialXScale = yRange / xRange;
        initialYScale = 1.0;
    }

    xScale.domain([initialXScale * xMin, initialXScale * xMax]);
    yScale.domain([initialYScale * yMin, initialYScale * yMax]);

    renderSpace = svg.append("g");
    renderSpace.attr("transform", "translate(" + marginLeft + "," + marginTop + ")");
    svg.call(zoom);

    // Add the X Axis
    xAxisNode = renderSpace.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + heightExMargin + ")")
        .style("stroke", gridColor)
        .style("stroke-width", "1.0")
        .style("fill", "none")
        .attr("font-size", textSize)
        .call(xAxis);
    xAxisNode.selectAll('text').style("stroke-width", 0).style("fill", "#000000");


    // Add the Y Axis
    yAxisNode = renderSpace.append("g")
        .attr("class", "y axis")
        .style("stroke", gridColor)
        .style("stroke-width", "1.0")
        .attr("font-size", textSize)
        .style("fill", "none")
        .call(yAxis);
    yAxisNode.selectAll('text').style("stroke-width", 0).style("fill", "#000000");


    // console.log("x/y min/max: " + xMin + ", " + xMax + ", " + yMin + ", " + yMax);

    //Add the data
    var data = x.map(function (d, i) {
        return {'xPos': x[i], 'yPos': y[i], 'name': name3[i]};
    });

    renderSpace.selectAll("circle")
        .data(data)
        .enter()
        .append("circle")
        .style("fill", "#000000")
        .attr("r", circleRadius)
        .attr("cx", function (d) {
            return xScale(d['xPos']);
        })
        .attr("cy", function (d) {
            return yScale(d['yPos']);
        });

    renderSpace.selectAll("text")
        .data(data)
        .enter()
        .append("text")
        .text(function (d) {
            return d['name'];
        })
        .attr("x", function (d) {
            return xScale(d['xPos']);
        })
        .attr("y", function (d) {
            return yScale(d['yPos']);
        })
        .attr("class", "tsneTextLabels")
        .attr("font-family", "sans-serif")
        .attr("font-size", textSize)
        .attr("fill", textColor);
}



function drawTsne() {
    if (currTsneSessionID == undefined) currTsneSessionID = "UploadedFile";


    $.ajax({
        url: "/tsne/coords/" + currTsneSessionID,
        cache: false
    })
        .done(function (data) {
            x = [];
            y = [];
            name3 = [];
            if (data.length > 0) {
                for (var i = 0; i < data.length; i++) {
                    var split = data[i].split(',');
                    var xCoord = parseFloat(split[0]);
                    var yCoord = parseFloat(split[1]);
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
            } else {
                $.notify({
                    title: '<strong>No t-SNE data available!</strong>',
                    message: 'Please upload something, or check again later.'
                }, {
                    type: 'danger',
                    placement: {
                        from: "top",
                        align: "center"
                    }
                });
            }
        });
}

function setSessionId(newID){
    currTsneSessionID = newID;
    drawTsne();
}

$(window).load(drawTsne());
