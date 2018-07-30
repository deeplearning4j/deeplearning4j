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

/*
    Here we receive simplified model description, and render it on page.
*/

var canvasWidth = 900;

var nodeWidth = 130;
var nodeHeight = 40;

var offsetVertical = 80;
var offsetHorizontal = 10;

var canvasLeft = 0;
var canvasTop = 0;
var canvasElements = [];
var lastNode = -1;

var margin = {top: 10, right: 30, bottom: 20, left: 30};
var width = 380 - margin.left - margin.right;
var height = 300 - margin.top - margin.bottom;

var marginFocus = {top: 10, right: 20, bottom: 100, left: 40};
var marginContext = {top: 270, right: 20, bottom: 20, left: 40};

var heightFocus = 300 - marginFocus.top - marginFocus.bottom;
var heightContext = 250 - marginContext.top - marginContext.bottom;


var gSVG = new Array();
var gXAxis = new Array();
var gYAxis = new Array();

var gX = new Array();
var gY = new Array();
var gXT = new Array();

var brush;
var focus;
var context;
var scoreData;
var area2;
var layerInit = new Array();

var init = 0;


var arrow = [
    [ 2, 0 ],
    [ -10, -4 ],
    [ -10, 4]
];

var scoreChart;

var layers = new Layers();

function drawFilledPolygon(ctx, shape) {
    ctx.beginPath();
    ctx.moveTo(shape[0][0],shape[0][1]);

    for(p in shape)
        if (p > 0) ctx.lineTo(shape[p][0],shape[p][1]);

    ctx.lineTo(shape[0][0],shape[0][1]);
    ctx.fill();
};

function translateShape(shape,x,y) {
    var rv = [];
    for(p in shape)
        rv.push([ shape[p][0] + x, shape[p][1] + y ]);
    return rv;
};

function rotateShape(shape,ang) {
    var rv = [];
    for(p in shape)
        rv.push(rotatePoint(ang,shape[p][0],shape[p][1]));
    return rv;
};
function rotatePoint(ang,x,y) {
    return [
        (x * Math.cos(ang)) - (y * Math.sin(ang)),
        (x * Math.sin(ang)) + (y * Math.cos(ang))
    ];
};

function drawLineArrow(ctx, x1,y1,x2,y2) {
    ctx.beginPath();
    ctx.moveTo(x1,y1);
    ctx.lineTo(x2,y2);
    ctx.stroke();
    var ang = Math.atan2(y2-y1,x2-x1);
    drawFilledPolygon(ctx, translateShape(rotateShape(arrow,ang),x2,y2));
};

function drawIntraLayerArrow(ctx, x1, y1, x2, y2) {
    ctx.beginPath();
    ctx.moveTo(x1,y1);
    ctx.bezierCurveTo(x1 + 10, y1 - (offsetVertical / 4), x2 - 10, y2 - (offsetVertical / 4), x2,y2);
    ctx.stroke();
    var ang = 0.9;
    drawFilledPolygon(ctx, translateShape(rotateShape(arrow,ang),x2,y2));
}

/*
    This method draws connections from each node, to all nodes it's connected to
*/
function renderConnections(ctx, layer) {
    for (var c = 0; c < layer.connections.length; c++) {
        var connection = layer.connections[c];
        if (connection.y == layer.y + 1) {
            // this is direct connection to the next layer, draw straight line
            var cX1 = getNodeX(layer.x, layer.y, layers.getLayersForY(layer.y).length) + (nodeWidth / 2);
            var cY1 = getNodeY(layer.x, layer.y) + nodeHeight + 5;

            var cX2 = getNodeX(connection.x, connection.y, layers.getLayersForY(connection.y).length)  + (nodeWidth / 2);
            var cY2 = getNodeY(connection.x, connection.y) - 5;

            drawLineArrow(ctx, cX1, cY1, cX2, cY2);
        } else if (connection.y == layer.y -1 ) {
            // this is direct connection to the previous layer, draw straight line
            var cX1 = getNodeX(layer.x, layer.y, layers.getLayersForY(layer.y).length) + (nodeWidth / 2);
            var cY1 = getNodeY(layer.x, layer.y) - 5;

            var cX2 = getNodeX(connection.x, connection.y, layers.getLayersForY(connection.y).length)  + (nodeWidth / 2);
            var cY2 = getNodeY(connection.x, connection.y) + nodeHeight + 5;

            drawLineArrow(ctx, cX1, cY1, cX2, cY2);
        } else if (connection.y == layer.y) {
            // this is connection withing same layer, bezier curve required
            if (layer.x < connection.x) {
                var cX1 = getNodeX(layer.x, layer.y, layers.getLayersForY(layer.y).length) + (nodeWidth / 2) + 20;
                var cY1 = getNodeY(layer.x, layer.y) - 5;

                var cX2 = getNodeX(connection.x, connection.y, layers.getLayersForY(layer.y).length) + (nodeWidth / 2) - 20;
                var cY2 = getNodeY(connection.x, connection.y) - 5;


                drawIntraLayerArrow(ctx, cX1, cY1, cX2, cY2);
            } else {
                var cX1 = getNodeX(layer.x, layer.y, layers.getLayersForY(layer.y).length) + (nodeWidth / 2) - 20;
                var cY1 = getNodeY(layer.x, layer.y) - 5;

                var cX2 = getNodeX(connection.x, connection.y, layers.getLayersForY(layer.y).length) + (nodeWidth / 2) + 20;
                var cY2 = getNodeY(connection.x, connection.y) - 5;


                drawIntraLayerArrow(ctx, cX1, cY1, cX2, cY2);
            }
        } else {
            // this is indirect connection, curve required
        }
    }
}

/*
    This method renders single specified grid
*/
function renderGrid(ctx, array, y) {
    var totalOnLayer = array.length;
    for (var x = 0; x < totalOnLayer; x++) {
        layer = layers.getLayerForYX(x, y);

        renderNode(ctx, layer, x, y, totalOnLayer );
        renderConnections(ctx, layer);
    }
}

/*
    This method returns proper X coordinates for specific X and Y position
*/
function getNodeX(x, y, totalOnLayer) {
    /*
        We want whole layer to be aligned to center nicely
    */

    var layerWidth = totalOnLayer * (nodeWidth + offsetHorizontal);
    var zeroX = (canvasWidth / 2) - (layerWidth / 2);

    var cX = zeroX + ((nodeWidth + offsetHorizontal) * x);

    return cX;
}

function getNodeY(x, y) {
    return  ((nodeHeight + offsetVertical) * y) + 1;
}

/*
    This method renders single node
*/
function renderNode(ctx, layer, x, y, totalOnLayer) {
    var cx = getNodeX(x, y, totalOnLayer);
    var cy = getNodeY(x, y);

    canvasElements.push({
        x: cx,
        y: cy,
        width: nodeWidth,
        height: nodeHeight,
        id: layer.id
    });

    // draw node rect
    ctx.beginPath();
    ctx.lineWidth = "1";
    ctx.fillStyle = layer.color;
    ctx.rect(cx, cy, nodeWidth, nodeHeight);
    ctx.fillRect(cx+1, cy+1, nodeWidth-2, nodeHeight-2);
//    console.log("cX: " + cx + " cY: " + cy + " width: " + nodeWidth + " height: " + nodeHeight);
    ctx.stroke();

    // draw description
    ctx.fillStyle = "#000000";
    ctx.font = "15px Roboto";
    ctx.textAlign="center";
    ctx.fillText(layer.layerType, cx + (nodeWidth / 2), cy + 25, nodeWidth - 10);

//    ctx.font = "12px Roboto";
//    ctx.fillText(layer.mainLine, cx + (nodeWidth / 2), cy + 45, (nodeWidth - 10));
//    ctx.font = "11px Roboto";
//    ctx.fillText(layer.subLine, cx + (nodeWidth / 2), cy + 70, (nodeWidth - 10));
}

function showView(id) {
    var layer = layers.getLayerForID(id);

    for (var i = 0; i < layers.layers.length; i++) {
        if (layers.layers[i].id != id)
            $("#panel_"+i).hide();
        else $("#panel_"+i).show();
    }
}

function renderLayers(container, layers) {
    $("#" + container).html("");

    // define grid parameters
    var canvasHeight = (nodeHeight * (layers.maximumY + 3)) + (offsetVertical * layers.maximumY) ;
    var canvasWidth = 900;

    $("#" + container).html("<canvas id='flowCanvas' width="+900+" height="+ canvasHeight +" >Canvas isn't supported on your browser</canvas>");

    var c=document.getElementById("flowCanvas");
    var ctx=c.getContext("2d");

    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    canvasLeft = c.offsetLeft,
    canvasTop = c.offsetTop,


    // to process clicks
    c.addEventListener('click', function(event) {
        var x = event.pageX - canvasLeft;
        var y = event.pageY - canvasTop;

        canvasElements.forEach(function(element) {
            if (y > element.y && y < element.y + element.height && x > element.x && x < element.x + element.width) {

                // here we go for element.id as active node
                $("#hint").hide();
                showView(element.id);
            }
        })
    }, false);

    // to process mouseovers
    c.addEventListener('mousemove', function(event) {
        var x = event.pageX - canvasLeft;
        var y = event.pageY - canvasTop;

        var got_something = false;

        canvasElements.forEach(function(element) {
            if (y > element.y && y < element.y + element.height && x > element.x && x < element.x + element.width) {
                // mouse is over element.id element
                got_something = true;
                $('#tooltip').css({left: event.pageX + 15, top: event.pageY + 15, opacity: 1});

                if (lastNode != element.id) {
                    var layer = layers.getLayerForID(element.id);
                    $('#tooltip').html(layer.description);
                }

                lastNode = element.id;
            }
        })

        if (got_something == false) {
            // hide tooltip
            lastNode = -1;
            $('#tooltip').css({opacity: 0});
        }

    }, false);


    for (var y = 0; y <= layers.maximumY; y++) {

        var yLayers = layers.getLayersForY(y);
        renderGrid(ctx, yLayers, y);
    }


}

function drawScores(values, id) {
        if (gSVG[id] != undefined || gSVG[id] != null) {
            var valueline = d3.svg.line()
                    .x(function(d,i) { return gX[id](i); })
                    .y(function(d) { return gY[id](d); });

            var max = d3.max(values);
            var min = d3.min(values);
            gX[id].domain([0,values.length]);
            gY[id].domain([min, max]);


            focus.select(".line")
                .attr("d", valueline(values));

            focus.select(".x.axis")
                .call(gXAxis[id]);

            focus.select(".y.axis")
                .call(gYAxis[id]);

            return;
        }

        gX[id] = d3.scale.linear().range([0, width]);
        gY[id] = d3.scale.linear().range([heightFocus, 0]);

        gX["context"] = d3.scale.linear().range([0, width]);
        gY["context"] = d3.scale.linear().range([heightContext, 0]);

        gSVG[id] = d3.select("#scoreChart")
                        .append("svg")
                        .attr("width", width + margin.left + margin.right)
                        .attr("height", height + margin.top + margin.bottom)
                        .append("g");

        focus = gSVG[id].append("g")
                    .attr("class", "focus")
                    .attr("transform", "translate(" + marginFocus.left + "," + marginFocus.top + ")");


        // Define the axes
        gXAxis[id]= d3.svg.axis().scale(gX[id])
        //               .innerTickSize(-heightFocus)     //used as grid line
                       .orient("bottom");//.ticks(5);

        gYAxis[id] = d3.svg.axis().scale(gY[id])
                   //     .innerTickSize(-width)      //used as grid line
                   .orient("left"); //.ticks(5);

        // Define the line
        var valueline = d3.svg.line()
                .x(function(d,i) { return gX[id](i); })
                .y(function(d) { return gY[id](d); });

        // Scale the range of the data
        var max = d3.max(values);
        var min = d3.min(values);
        gX[id].domain([0,values.length]);
        gY[id].domain([min, max]);

        // Add the valueline path.
        focus.append("path")
                        .attr("class", "line")
                        .attr("d",  valueline(values));


        // Add the X Axis
        focus.append("g")
                        .attr("class", "x axis")
                        .attr("transform", "translate(0," + heightFocus + ")")
                        .call(gXAxis[id]);

        // Add the Y Axis
        focus.append("g")
                        .attr("class", "y axis")
                        .call(gYAxis[id]);



}

function drawPerf(samples, batches, time) {
    var fixed_samples = parseFloat(samples).toFixed(2);
    var fixed_batches = parseFloat(batches).toFixed(2);

    $("#ps").html("" + fixed_samples + "/sec");
    $("#pb").html("" + fixed_batches + "/sec");
    $("#pt").html("" + time + " ms");
}

function drawParams(values, layer, set) {
    if (init < 1)
        return;

    var formatCount = d3.format(",.0f");
    var data = [];
    var binNum = 0;
    var binTicks = [];
    var min = null;
    var max = null;

    // convert json to d3 data structure
    var keys = Object.keys(values);
    for (var k = 0; k < keys.length; k++) {
        var key = keys[k];
        var fkey = parseFloat(key);
        var value = parseInt(values[key]);

        if (min == null) min = fkey;
        if (max == null) max = fkey;

        if (min > fkey) min = fkey;
        if (max < fkey) max = fkey;


        data.push({"x": parseFloat(key), "y": value});
        binTicks.push(key);
        binNum++;
    }

    var binWidth = parseFloat(width / (binNum - 1)) - 1;

    var id = "id_" + layer + "_" + set;
    var selector = "#view_" + id;

    if (init == 2) {
        // update charts
        if (gSVG[id] == undefined || gSVG[id] == null) {
            console.log("skipping id: " + id);
            return;
        }


        console.log("Updating: " + id);

        gX[id] = d3.scale.linear()
                        .domain([min, max])
                        .range([0, width]);

        gXT[id] = d3.scale.linear()
                        .domain([min, max])
                        .range([0, width - margin.right - 5]);

        gY[id] = d3.scale.linear()
                        .domain([0, d3.max(data, function(d) { return d.y; })])
                        .range([height, 0]);

        gXAxis[id] = d3.svg.axis()
                        .scale(gX[id])
                        .orient("bottom")
                        .tickValues(binTicks);


        var bar = gSVG[id].selectAll(".bar")
                        .data(data)
                        .attr("transform", function(d) { return "translate(" + gXT[id](d.x) + "," + gY[id](d.y) + ")"; });

        gSVG[id].selectAll("text")
                        .data(data)
                        .attr("y", 6)
                        .text(function(d) { return formatCount(d.y); });

        gSVG[id].selectAll("rect")
                        .data(data)
                        .attr("y", function(d) {
                            return 0;
                        })
                        .attr("height", function(d) { return height-gY[id](d.y) });

        gSVG[id].selectAll(".x.axis")
                        .attr("transform", "translate(0," + height + ")")
                        .call(gXAxis[id]);

        return;
    }

    // create charts
    $("#holder_" + layer).html("");


    gX[id] = d3.scale.linear()
                     .domain([min, max])
                     .range([0, width]);

    gXT[id] = d3.scale.linear()
                      .domain([min, max])
                      .range([0, width - margin.right + 5]);

    gY[id] = d3.scale.linear()
                    .domain([0, d3.max(data, function(d) { return d.y; })])
                    .range([height, 0]);



    gSVG[id] = d3.select(selector).append("svg")
                    .attr("width", width + margin.left + margin.right)
                    .attr("height", height + margin.top + margin.bottom)
                    .append("g")
                    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    gXAxis[id] = d3.svg.axis()
                    .scale(gX[id])
                    .orient("bottom")
                    .tickValues(binTicks);

    var bar = gSVG[id].selectAll(".bar")
                    .data(data)
                    .enter()
                    .append("g")
                    .attr("class", "bar")
                    .attr("transform", function(d) { return "translate(" + gXT[id](d.x) + "," + gY[id](d.y) + ")"; });

    bar.append("rect")
                    .attr("x", 1)
                    .attr("y", 0)
                    .attr("width", binWidth - 3)
                    .attr("height", function(d) {
                            return height - gY[id](d.y);
                            });

    bar.append("text")
                    .attr("dy", ".75em")
                    .attr("y", 6)
                    .attr("x", binWidth - (binWidth / 2))
                    .attr("text-anchor", "middle")
                    .attr("color","#000000")
                    .attr("font-size","9px")
                    .text(function(d) { return formatCount(d.y); });

    gSVG[id].append("g")
                    .attr("class", "x axis")
                    .attr("transform", "translate(0," + height + ")")
                    .call(gXAxis[id]);
}

function stateFunction() {
    var sid = getParameterByName("sid");
    if (sid == undefined) sid = 0;

    $.ajax({
        url:"/flow" + "/state?sid=" + sid,
        async: true,
        error: function (query, status, error) {
            $.notify({
                title: '<strong>No connection!</strong>',
                message: 'DeepLearning4j UiServer seems to be down!'
            },{
                type: 'danger',
                placement: {
                    from: "top",
                    align: "center"
                },
            });
            setTimeout(stateFunction, 10000);
        },
        success: function( data ) {
            //
            var scores = data['scores'];
            var score = parseFloat(data['score']).toFixed(5);
            var samples = data['performanceSamples'];
            var batches = data['performanceBatches'];
            var time = data['iterationTime'];
            var timeSpent = data['trainingTime'];
            var params = data['layerParams'];
            var lrs = data['learningRates'];


            drawPerf(samples, batches, time);
            drawScores(scores, "scores");


            $("#ss").html(""+ score);
            $("#st").html(timeSpent);

            if (init > 0) {
                var paramsKeys = Object.keys(params);

                if (paramsKeys == undefined || paramsKeys.length < 1)
                    return;

                for (var i = 0; i < lrs.length; i++) {
                    $('#lr_'+(i+1)).html(lrs[i]);
                }

                // we iterate over layers as keys
                for (var key in params) {
                    var layerParams = params[key];

                    // we have to create holders for histograms PRIOR to svg initialization
                    if (init < 2) {
                        for (var set in layerParams) {
                            if (layerParams[set] == null || layerParams[set] == undefined)
                                continue;

                            var description = "";
                            if (set === "w") {
                                description = "Weights:";
                            } else if (set === "rw") {
                                description = "Recurrent weights:";
                            } else if (set === "b") {
                                description = "Biases:";
                            } else if (set === "rwf") {
                                description = "Recurrent weights forward:";
                            }

                            var layer = parseInt(key) + 1;
                            var id = "id_" + layer + "_" + set;
                            var html = $("#panel_"+ layer).html() +"<b>"+ description +"</b><br/>" + "<div id='view_"+id+"'></div>"
                            $("#panel_"+ layer).html(html);
                        }
                    }

                    // each layer may have multiple param sets: W, RW, RWF, B
                    for (var set in layerParams) {

                        // if some specifc set, ie RWF is null - just skip it
                        if (layerParams[set] == null || layerParams[set] == undefined)
                            continue;

                        var data = layerParams[set];

                        drawParams(data, parseInt(key) + 1, set);
                    }
                }

                init = 2;
            }



            setTimeout(stateFunction, 2000);
        }
    });
}

function timedFunction() {

     var sid = getParameterByName("sid");
     if (sid == undefined) sid = 0;

     $.ajax({
                            url:"/flow" + "/info?sid=" + sid,
                            async: true,
                            error: function (query, status, error) {
                                $.notify({
                                    title: '<strong>No connection!</strong>',
                                    message: 'DeepLearning4j UiServer seems to be down!'
                                },{
                                    type: 'danger',
                                    placement: {
                                        from: "top",
                                        align: "center"
                                        },
                                });
                                setTimeout(timedFunction, 5000);
                            },
                            success: function( data ) {
                                // parse & render ModelInfo
                                //console.log("data received: " + data);

                                var updateTime = data['time'];
                                var time = new Date(updateTime);
                                $('#updatetime').html(time.customFormat("#DD#/#MM#/#YYYY# #hhh#:#mm#:#ss#"));

                                if (typeof data.layers == 'undefined') return;

                                layers = new Layers();

                                /*
                                    At this point we're going to have array of objects, with some properties tied.
                                    Rendering will be using pseudo-grid, derived from original layers connections
                                */
                                var html = "<div style='position: relative; top: 45%; height: 40px; margin: 0 auto;' id='hint'><b>&lt; Click on any node for detailed report</b></div>";
                                for (var i = 0; i < data.layers.length; i++) {
                                    var layer = new Layer(data.layers[i]);
                                    layers.attach(layer);

                                    html += "<div id='panel_"+layer.id+"' style='display: none;'><p class='layerDesc'>Layer name: <b>"+ layer.name+ "</b><br/>Layer type: " + layer.layerType+ "<br/>" + layer.description+ "Learning rate: <span id='lr_"+layer.id+"'>N/A</span></p><br/><div id='holder_"+layer.id+"'><b>No parameters available for this node</b></div></div>";
                                }
                                $("#viewport").html(html);
                                renderLayers("display", layers);

                                init = 1;
                            }
              });

}


setTimeout(timedFunction,2000);
setTimeout(stateFunction,2000);
$(window).resize(timedFunction);