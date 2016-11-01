/* ---------- Mean Magnitudes Chart ---------- */
function renderMeanMagChart() {

    $.ajax({
        url: "/train/model/data/0",
        async: true,
        error: function (query, status, error) {
            console.log("Error getting data: " + error);
        },
        success: function (data) {
            console.log("Keys: " + Object.keys(data));

            var b = data["meanMagRatio"]["b"];
            var w = data["meanMagRatio"]["w"];
            var iter = data["meanMagRatio"]["iterCounts"];


            if ($("#scoreiterchart").length) {
                var b = [];
                var w = [];

                for (var i = 0; i < iter.length; i++) {
                    b.push([iter[i], b[i]]);
                    w.push([iter[i], w[i]]);
                }

                var plot = $.plot($("#scoreiterchart"),
                    [{data: b, label: "b"},{data: w, label: "w"}], {
                        series: {
                            lines: {
                                show: true,
                                lineWidth: 2,
                            },
                            points: {show: true},
                            shadowSize: 2
                        },
                        grid: {
                            hoverable: true,
                            clickable: true,
                            tickColor: "#dddddd",
                            borderWidth: 0
                        },
                        yaxis: {min: 0, max: maxScore},
                        colors: ["#FA5833", "#2FABE9"]
                    });

                var previousPoint = null;
                $("#scoreiterchart").bind("plothover", function (event, pos, item) {
                    $("#x").text(pos.x.toFixed(0));
                    $("#y").text(pos.y.toFixed(2));
                });
            }
        }
    });

}

/* ---------- Layer Table Data ---------- */
function renderLayerTable() {

    $.ajax({
        url: "/train/model/graph",
        async: true,
        error: function (query, status, error) {
            console.log("Error getting data: " + error);
        },
        success: function (data) {
            console.log("Keys: " + Object.keys(data));

            /* Layer */
            var layerName = data["layerNames"][1];
            var layerType = data["layerTypes"][1];
            var inputSize = data["layerInfo"]["Input size"];
            var outputSize = data["layerInfo"]["Output size"];
            var nParams = data["layerInfo"]["Num Parameters"];
            var activationFunction = data["layerInfo"]["Activation Function"];
            var lossFunction = data["layerInfo"]["Loss Function"];

            $("#layerName").html(layerName);
            $("#layerType").html(layerType);
            $("#inputSize").html(inputSize);
            $("#outputSize").html(outputSize);
            $("#nParams").html(nParams);
            $("#activationFunction").html(activationFunction);
            $("#lossFunction").html(lossFunction);
        }
    });

}