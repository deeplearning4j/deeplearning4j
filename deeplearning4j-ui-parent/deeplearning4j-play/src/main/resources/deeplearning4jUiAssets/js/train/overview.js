/* ---------- Variances chart selection ---------- */
var selectedChart = "stdevActivations";
function selectStdevChart(fieldName) {
    selectedChart = fieldName;
    lastUpdateTime = -1;    //Reset update time to force reload

    //Tab highlighting logic
    if (selectedChart == "stdevActivations") {
        $("#stdevActivations").attr("class", "active");
        $("#stdevGradients").removeAttr("class");
        $("#stdevUpdates").removeAttr("class");
    }
    else if (selectedChart == "stdevGradients") {
        $("#stdevActivations").removeAttr("class");
        $("#stdevGradients").attr("class", "active");
        $("#stdevUpdates").removeAttr("class");
    }
    else {
        $("#stdevActivations").removeAttr("class");
        $("#stdevGradients").removeAttr("class");
        $("#stdevUpdates").attr("class", "active");
    }
}

/* ---------- Render page ---------- */
var lastUpdateTime = -1;
var lastUpdateSession = "";
function renderOverviewPage(firstLoad) {
    updateSessionWorkerSelect();

    if(firstLoad || !lastUpdateSession || lastUpdateSession == "" || lastUpdateSession != currSession){
        executeOverviewUpdate();
    } else {
        //Check last update time first - see if data has actually changed...
        $.ajax({
            url: "/train/sessions/lastUpdate/" + currSession,
            async: true,
            error: function (query, status, error) {
                console.log("Error getting data: " + error);
            },
            success: function (data) {
                if(data > lastUpdateTime){
                    executeOverviewUpdate();
                }
            }
        });
    }
}

function executeOverviewUpdate(){
    $.ajax({
        url: "/train/overview/data",
        async: true,
        error: function (query, status, error) {
            console.log("Error getting data: " + error);
        },
        success: function (data) {
            lastUpdateSession = currSession;
            lastUpdateTime = data["updateTimestamp"];
            renderScoreVsIterChart(data);
            renderModelPerformanceTable(data);
            renderUpdatesRatio(data);
            renderStdevChart(data);
        }
    });
}

/* ---------- Score vs. Iteration Chart ---------- */
function renderScoreVsIterChart(data) {
    var scoresArr = data["scores"];
    var scoresIter = data["scoresIter"];

    var maxScore = Math.max.apply(Math, scoresArr);
    var chartMin = Math.min.apply(Math, scoresArr);
    if(chartMin > 0){
        chartMin = 0.0;
    }

    var scoreChart = $("#scoreiterchart");
    scoreChart.unbind(); // prevent over-subscribing

    if (scoreChart.length) {
        var scoreData = [];

        for (var i = 0; i < scoresArr.length; i++) {
            scoreData.push([scoresIter[i], scoresArr[i]]);
        }

        var plotData = [{data: scoreData, label: "score"}];

        // calculate a EMA line to summarize training progress
        if(scoresIter.length > 10) {
            var bestFitLine = EMACalc(scoresArr, 10);
            var bestFitData = [];
            for (var i = 0; i < bestFitLine.length; i++) {
                bestFitData.push([scoresIter[i], bestFitLine[i]]);
            }
            plotData.push({data: bestFitData, label: "summary"});
        }

        // plot the chart
        var plotOptions = {
            series: {
                lines: {
                    show: true,
                    lineWidth: 2
                }
            },
            grid: {
                hoverable: true,
                clickable: true,
                tickColor: "#dddddd",
                borderWidth: 0
            },
            yaxis: {min: chartMin, max: maxScore},
            colors: ["#FA5833","rgba(65,182,240,0.3)","#000000"],
            selection: {
                mode: "x"
            }
        };
        var plot = $.plot(scoreChart, plotData, plotOptions);

        // when selected, calculate best fit line
        scoreChart.bind("plotselected", function (event, ranges) {
            var indices = [];
            var fromIdx = parseInt(ranges.xaxis.from);
            var toIdx = parseInt(ranges.xaxis.to);
            var scoresCopy = scoresArr.slice();

            for (var i = fromIdx; i <= toIdx; i++) {
               indices.push(i);
            }

            var bestFitLine = findLineByLeastSquares(indices, scoresCopy.slice(fromIdx,toIdx+1));
            var bestFitData = [];
            for (var i = 0; i < bestFitLine[0].length; i++) {
                bestFitData.push([bestFitLine[0][i], bestFitLine[1][i]]);
            }
            plotData.push({data: bestFitData, label: "selection"});

            plot.setData(plotData);
            plot.draw();
        });

        scoreChart.bind("plotunselected", function (event) {
            plotData = plotData.slice(0,2);
            plot.setData(plotData);
            plot.draw();
        });

        function showTooltip(x, y, contents) {
            $('<div id="tooltip">' + contents + '</div>').css({
                position: 'absolute',
                display: 'none',
                top: y + 8,
                left: x + 10,
                border: '1px solid #fdd',
                padding: '2px',
                'background-color': '#dfeffc',
                opacity: 0.80
            }).appendTo("#scoreiterchart").fadeIn(200);
        }

        var previousPoint = null;
        scoreChart.bind("plothover", function (event, pos, item) {
            if (typeof pos.x == 'undefined') return;

            var xPos = pos.x.toFixed(0);
            $("#x").text(xPos < 0 || xPos == "-0" ? "" : xPos);
            $("#y").text(pos.y.toFixed(5));

            if (item) {
                if (previousPoint != item.dataIndex) {
                    previousPoint = item.dataIndex;

                    $("#tooltip").remove();
                    var x = item.datapoint[0].toFixed(0);
                    var y = item.datapoint[1].toFixed(5);

                    showTooltip(item.pageX - scoreChart.offset().left, item.pageY - scoreChart.offset().top,
                        "(" + x + ", " + y + ")");
                }
            }
            else {
                $("#tooltip").remove();
                previousPoint = null;
            }
        });
    }
}

/* ---------- Model Performance Table ---------- */
function renderModelPerformanceTable(data) {

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

    /* Inject Performance Information */
    $("#startTime").html(startTime);
    $("#totalRuntime").html(totalRuntime);
    $("#lastUpdate").html(lastUpdate);
    $("#totalParamUpdates").html(totalParamUpdates);
    $("#updatesPerSec").html(updatesPerSec);
    $("#examplesPerSec").html(examplesPerSec);
}

/* ---------- Ratio of Updates to Parameters Chart ---------- */
function renderUpdatesRatio(data) {
    var ratios = data["updateRatios"];

    var iter = data["scoresIter"];

    var chart = $("#updateRatioChart");

    if (chart.length) {

        var keys = Object.keys(ratios);
        var toPlot = [];
        var overallMax = -Number.MAX_VALUE;
        var overallMin = Number.MAX_VALUE;
        for (var i = 0; i < keys.length; i++) {
            var r = ratios[keys[i]];

            var pairs = [];
            for (var j = 0; j < r.length; j++) {
                pairs.push([iter[j], Math.log10(r[j])]);
            }
            toPlot.push({data: pairs, label: keys[i]});


            var thisMax = Math.max.apply(Math, r);
            var thisMin = Math.min.apply(Math, r);
            overallMax = Math.max(overallMax, thisMax);
            overallMin = Math.min(overallMin, thisMin);
        }

        if (overallMax == -Number.MAX_VALUE) overallMax = 1.0;
        if (overallMin == Number.MAX_VALUE) overallMin = 0.0;

        overallMax = Math.log10(overallMax);
        overallMin = Math.log10(overallMin);
        overallMin = Math.max(overallMin, -10);

        overallMax = Math.ceil(overallMax);
        overallMin = Math.floor(overallMin);

        var plot = $.plot(chart,
            toPlot, {
                series: {
                    lines: {
                        show: true,
                        lineWidth: 2
                    }
                    // points: {show: true},
                    // shadowSize: 2
                },
                grid: {
                    hoverable: true,
                    clickable: true,
                    tickColor: "#dddddd",
                    borderWidth: 0
                },
                yaxis: {min: overallMin, max: overallMax},
                colors: ["#FA5833", "#2FABE9"]
            });


        function showTooltip(x, y, contents) {
            $('<div id="tooltipRatioChart">' + contents + '</div>').css({
                position: 'absolute',
                display: 'none',
                top: y + 8,
                left: x + 10,
                border: '1px solid #fdd',
                padding: '2px',
                'background-color': '#dfeffc',
                opacity: 0.80
            }).appendTo("#updateRatioChart").fadeIn(200);
        }

        var previousPoint = null;
        chart.bind("plothover", function (event, pos, item) {
            if (typeof pos.x == 'undefined') return;

            var xPos = pos.x.toFixed(0);
            $("#xRatio").text(xPos < 0 || xPos == "-0" ? "" : xPos);
            $("#yLogRatio").text(pos.y.toFixed(5));
            $("#yRatio").text(Math.pow(10, pos.y).toFixed(5));

            if (item) {
                if (previousPoint != item.dataIndex) {
                    previousPoint = item.dataIndex;

                    $("#tooltipRatioChart").remove();
                    var x = item.datapoint[0].toFixed(0);
                    var logy = item.datapoint[1].toFixed(5);
                    var y = Math.pow(10, item.datapoint[1]).toFixed(5);

                    showTooltip(item.pageX - chart.offset().left, item.pageY - chart.offset().top,
                        "(" + x + ", logRatio=" + logy + ", ratio=" + y + ")");
                }
            }
            else {
                $("#tooltipRatioChart").remove();
                previousPoint = null;
            }
        });
    }
}


/* ---------- Stdev Charts ---------- */
function renderStdevChart(data) {
    var selected = selectedChart;
    var chart = $("#stdevChart");

    if (chart.length) {

        var stdevs = data[selected];
        var iter = data["scoresIter"];
        var keys = Object.keys(stdevs);

        var toPlot = [];
        var overallMax = -Number.MAX_VALUE;
        var overallMin = Number.MAX_VALUE;
        for (var i = 0; i < keys.length; i++) {
            var r = stdevs[keys[i]];

            var pairs = [];
            for (var j = 0; j < r.length; j++) {
                pairs.push([iter[j], Math.log10(r[j])]);
            }
            toPlot.push({data: pairs, label: keys[i]});


            var thisMax = Math.max.apply(Math, r);
            var thisMin = Math.min.apply(Math, r);
            overallMax = Math.max(overallMax, thisMax);
            overallMin = Math.min(overallMin, thisMin);
        }

        if (overallMax == -Number.MAX_VALUE) overallMax = 1.0;
        if (overallMin == Number.MAX_VALUE) overallMin = 0.0;

        overallMax = Math.log10(overallMax);
        overallMin = Math.log10(overallMin);
        overallMin = Math.max(overallMin, -10);

        overallMax = Math.ceil(overallMax);
        overallMin = Math.floor(overallMin);


        var plot = $.plot(chart,
            toPlot, {
                series: {
                    lines: {
                        show: true,
                        lineWidth: 2
                    }
                },
                grid: {
                    hoverable: true,
                    clickable: true,
                    tickColor: "#dddddd",
                    borderWidth: 0
                },
                yaxis: {min: overallMin, max: overallMax},
                colors: ["#FA5833", "#2FABE9"]
            });


        function showTooltip(x, y, contents) {
            $('<div id="tooltipStdevChart">' + contents + '</div>').css({
                position: 'absolute',
                display: 'none',
                top: y + 8,
                left: x + 10,
                border: '1px solid #fdd',
                padding: '2px',
                'background-color': '#dfeffc',
                opacity: 0.80
            }).appendTo("#stdevChart").fadeIn(200);
        }

        var previousPoint = null;
        chart.bind("plothover", function (event, pos, item) {
            if (typeof pos.x == 'undefined') return;

            var xPos = pos.x.toFixed(0);
            $("#xStdev").text(xPos < 0 || xPos == "-0" ? "" : xPos);
            $("#yLogStdev").text(pos.y.toFixed(5));
            $("#yStdev").text(Math.pow(10, pos.y).toFixed(5));

            //Tooltip
            if (item) {
                if (previousPoint != item.dataIndex) {
                    previousPoint = item.dataIndex;

                    $("#tooltipStdevChart").remove();
                    var x = item.datapoint[0].toFixed(0);
                    var logy = item.datapoint[1].toFixed(5);
                    var y = Math.pow(10, item.datapoint[1]).toFixed(5);

                    showTooltip(item.pageX - chart.offset().left, item.pageY - chart.offset().top,
                        item.series.label + " (" + x + ", logStdev=" + logy + ", stdev=" + y + ")");
                }
            }
            else {
                $("#tooltipStdevChart").remove();
                previousPoint = null;
            }
        });
    }
}

/* --------------- linear least squares (best fit line) ---------- */
function findLineByLeastSquares(values_x, values_y) {
    var sum_x = 0;
    var sum_y = 0;
    var sum_xy = 0;
    var sum_xx = 0;
    var count = 0;

    /*
     * We'll use those variables for faster read/write access.
     */
    var x = 0;
    var y = 0;
    var values_length = values_x.length;

    if (values_length != values_y.length) {
        throw new Error('The parameters values_x and values_y need to have same size!');
    }

    /*
     * Nothing to do.
     */
    if (values_length === 0) {
        return [ [], [] ];
    }

    /*
     * Calculate the sum for each of the parts necessary.
     */
    for (var v = 0; v < values_length; v++) {
        x = values_x[v];
        y = values_y[v];
        sum_x += x;
        sum_y += y;
        sum_xx += x*x;
        sum_xy += x*y;
        count++;
    }

    /*
     * Calculate m and b for the formular:
     * y = x * m + b
     */
    var m = (count*sum_xy - sum_x*sum_y) / (count*sum_xx - sum_x*sum_x);
    var b = (sum_y/count) - (m*sum_x)/count;

    /*
     * We will make the x and y result line now
     */
    var result_values_x = [];
    var result_values_y = [];

    for (var v = 0; v < values_length; v++) {
        x = values_x[v];
        y = x * m + b;
        result_values_x.push(x);
        result_values_y.push(y);
    }

    return [result_values_x, result_values_y];
}


/* --------------- exponential moving average (best fit line) ---------- */
function EMACalc(mArray,mRange) {
  var k = 2/(mRange + 1);
  // first item is just the same as the first item in the input
  emaArray = [mArray[0]];
  // for the rest of the items, they are computed with the previous one
  for (var i = 1; i < mArray.length; i++) {
    emaArray.push(mArray[i] * k + emaArray[i - 1] * (1 - k));
  }
  return emaArray;
}
