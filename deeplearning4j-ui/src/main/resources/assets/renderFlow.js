/*
    Here we receive simplified model description, and render it on page.
*/

var nodeWidth = 150;
var nodeHeight = 100;

var offsetVertical = 100;
var offsetHorizontal = 10;

// canvas width
var width = 900;

var arrow = [
    [ 2, 0 ],
    [ -10, -4 ],
    [ -10, 4]
];

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
    var zeroX = (width / 2) - (layerWidth / 2);

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
    ctx.fillText(layer.name, cx + (nodeWidth / 2), cy + 20, nodeWidth - 10);

    ctx.font = "13px Roboto";
    ctx.fillText(layer.mainLine, cx + (nodeWidth / 2), cy + 45, (nodeWidth - 10));
    ctx.fillText(layer.subLine, cx + (nodeWidth / 2), cy + 70, (nodeWidth - 10));
}

function renderLayers(container, layers) {
    $("#" + container).html("");

    // define grid parameters
    var canvasHeight = (nodeHeight * (layers.maximumY + 3)) + (offsetVertical * layers.maximumY) ;

    $("#" + container).html("<canvas id='flowCanvas' width=900 height="+ canvasHeight +" >Canvas isn't supported on your browser</canvas>");

    var c=document.getElementById("flowCanvas");
    var ctx=c.getContext("2d");

    for (var y = 0; y <= layers.maximumY; y++) {

        var yLayers = layers.getLayersForY(y);
        renderGrid(ctx, yLayers, y);
    }


}

function timedFunction() {

     $.ajax({
                            url:"/flow" + "/state",
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

                                if (typeof data.layers == 'undefined') return;

                                layers = new Layers();

                                /*
                                    At this point we're going to have array of objects, with some properties tied.
                                    Rendering will be using pseudo-grid, derived from original layers connections
                                */
                                for (var i = 0; i < data.layers.length; i++) {
                                    var layer = new Layer(data.layers[i]);
                                    layers.attach(layer);
                                }
                                renderLayers("display", layers);
                            }
              });

}

//$(window).load(setTimeout(timedFunction,2000));
setTimeout(timedFunction,2000)