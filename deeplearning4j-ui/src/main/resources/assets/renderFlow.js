/*
    Here we receive simplified model description, and render it on page
*/

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
                                console.log("data received");
                            }
              });
}

$(window).load(setTimeout(timedFunction,2000));
