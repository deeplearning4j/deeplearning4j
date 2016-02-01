/*
    Here we receive simplified model description, and render it on page.
*/
var layers = [];
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
                                setTimeout(timed, 5000);
                            },
                            success: function( data ) {
                                // parse & render ModelInfo
                                console.log("data received: " + data);
                                /*
                                    At this point we're going to have array of objects, with some properties tied.
                                    Rendering will be using pseudo-grid, derived from original layers connections
                                */
                                for (var i = 0; i < data.layers.length; i++) {

                                    var layer = new Layer(data.layers[i]);
                                    layers.push(layer);
                                    console.log("Adding layer: " + layer.name + " ML: " + layer.mainLine);
                                }
                            }
              });

}

//$(window).load(setTimeout(timedFunction,2000));
setTimeout(timedFunction,2000)