/* ---------- Score vs. Iteration Chart ---------- */
function renderScoreChart() {

    $.ajax({
        url: "/train/overview/data",
        async: true,
        error: function (query, status, error) {
            console.log("Error getting data: " + error);
        },
        success: function (data) {
            console.log("Keys: " + Object.keys(data));

            var scoresArr = data["scores"];
            var scoresIter = data["scoresIter"];

            var maxScore = Math.max.apply(Math, scoresArr);


            if ($("#scoreiterchart").length) {
                var scoreData = [];

                for (var i = 0; i < scoresArr.length; i++) {
                    scoreData.push([scoresIter[i], scoresArr[i]]);
                }

                var plot = $.plot($("#scoreiterchart"),
                    [{data: scoreData, label: "Scores"}], {
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

/* ---------- Model Performance Table Data ---------- */
function renderPerformanceTable() {

    $.ajax({
        url: "/train/overview/data",
        async: true,
        error: function (query, status, error) {
            console.log("Error getting data: " + error);
        },
        success: function (data) {
            console.log("Keys: " + Object.keys(data));

            /* Model */
            var modelType = data["model"][0][1];
            var nLayers = data["model"][1][1];
            var nParams = data["model"][2][1];

            /* Performance */
            var startTime = data["perf"][0][1];
            var totalRuntime = data["perf"][1][1];
            var lastUpdate = data["perf"][2][1];
            var totalParamUpdates = data["perf"][3][1];
            var updatesPerSec = data["perf"][4][1];
            var examplesPerSec = data["perf"][5][1];

            /* Inject Model Information */
            $("#modelType").html(modelType);
            $("#nLayers").html(nLayers);
            $("#nParams").html(nParams);

            /* Inject Performance Information*/
            $("#startTime").html(startTime);
            $("#totalRuntime").html(totalRuntime);
            $("#lastUpdate").html(lastUpdate);
            $("#totalParamUpdates").html(totalParamUpdates);
            $("#updatesPerSec").html(updatesPerSec);
            $("#examplesPerSec").html(examplesPerSec);
        }
    });

}