function renderOverviewPage() {

    $.ajax({
        url: "/train/overview/data",
        async: true,
        error: function (query, status, error) {
            console.log("Error getting data: " + error);
        },
        success: function (data) {
            console.log("Keys: " + Object.keys(data));


            renderScoreVsIterChart(data);
            renderModelPerformanceTable(data);
            renderUpdatesRatio(data);
            renderVariancesRatio(data);
        }
    });

}


/* ---------- Score vs. Iteration Chart ---------- */
function renderScoreVsIterChart(data) {
    var scoresArr = data["scores"];
    var scoresIter = data["scoresIter"];

    var maxScore = Math.max.apply(Math, scoresArr);

    var scoreChart = $("#scoreiterchart");

    if (scoreChart.length) {
        var scoreData = [];

        for (var i = 0; i < scoresArr.length; i++) {
            scoreData.push([scoresIter[i], scoresArr[i]]);
        }

        var plot = $.plot($("#scoreiterchart"),
            [{data: scoreData, label: "score"}], {
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
                yaxis: {min: 0, max: maxScore},
                colors: ["#FA5833", "#2FABE9"]
            });

        function showTooltip(x, y, contents) {
            $('<div id="tooltip">' + contents + '</div>').css( {
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
            $("#x").text(pos.x.toFixed(0));
            $("#y").text(pos.y.toFixed(2));

            if (item) {
                if (previousPoint != item.dataIndex) {
                    previousPoint = item.dataIndex;

                    $("#tooltip").remove();
                    var x = item.datapoint[0].toFixed(0),
                        y = item.datapoint[1].toFixed(5);

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

    console.log("Update ratio keys: " + Object.keys(ratios));

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

        // console.log("Overall max/min: " + overallMax + "\t" + overallMin);

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
            $("#xRatio").text(pos.x.toFixed(0));
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




/* ---------- Variance Charts ---------- */
function renderVariancesRatio(data) {
    var selected = "varianceActivations";   //TODO: selection
    var chart = $("#varianceChart");

    if (chart.length) {

        var variances = data[selected];
        var iter = data["scoresIter"];
        var keys = Object.keys(variances);

        var toPlot = [];
        var overallMax = -Number.MAX_VALUE;
        var overallMin = Number.MAX_VALUE;
        for (var i = 0; i < keys.length; i++) {
            var r = variances[keys[i]];

            var pairs = [];
            for (var j = 0; j < r.length; j++) {
                // pairs.push([iter[j], r[j]]);
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
            $('<div id="tooltipVarianceChart">' + contents + '</div>').css({
                position: 'absolute',
                display: 'none',
                top: y + 8,
                left: x + 10,
                border: '1px solid #fdd',
                padding: '2px',
                'background-color': '#dfeffc',
                opacity: 0.80
            }).appendTo("#varianceChart").fadeIn(200);
        }

        var previousPoint = null;
        chart.bind("plothover", function (event, pos, item) {
            $("#xVariance").text(pos.x.toFixed(0));
            $("#yLogVariance").text(pos.y.toFixed(5));
            $("#yVariance").text(Math.pow(10,pos.y).toFixed(5));

            //Tooltip
            if (item) {
                if (previousPoint != item.dataIndex) {
                    previousPoint = item.dataIndex;

                    $("#tooltipVarianceChart").remove();
                    var x = item.datapoint[0].toFixed(0);
                    var logy = item.datapoint[1].toFixed(5);
                    var y = Math.pow(10, item.datapoint[1]).toFixed(5);

                    showTooltip(item.pageX - chart.offset().left, item.pageY - chart.offset().top,
                        item.series.label + " (" + x + ", logVariance=" + logy + ", variance=" + y + ")");
                }
            }
            else {
                $("#tooltipVarianceChart").remove();
                previousPoint = null;
            }
        });
    }
}

/* ---------- Language Dropdown ---------- */

	$('.dropmenu').click(function(e){
		e.preventDefault();
		$(this).parent().find('ul').slideToggle();
	});
