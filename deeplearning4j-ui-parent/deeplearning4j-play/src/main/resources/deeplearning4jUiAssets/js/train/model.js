
var selectedVertex = 0;

function setSelectedVertex(vertex){
    selectedVertex = vertex;
}


function renderModelPage() {

    console.log("Currently selected vertex: " + selectedVertex);

    $.ajax({
        url: "/train/model/data/" + selectedVertex,
        async: true,
        error: function (query, status, error) {
            console.log("Error getting data: " + error);
        },
        success: function (data) {
            console.log("Keys: " + Object.keys(data));

            renderMeanMagChart(data);
            renderActivationsChart(data);
            renderLearningRateChart(data);
            renderParametersHistogram(data);
        }
    });

}

/* ---------- Mean Magnitudes Chart ---------- */
function renderMeanMagChart(data) {

    var b = data["meanMagRatio"]["b"];
    var W = data["meanMagRatio"]["W"];
    var iter = data["meanMagRatio"]["iterCounts"];

    if ($("#meanmag").length) {

        var ratios = data["meanMagRatio"]["ratios"];
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

        // console.log("MM CHART: " + overallMin + "\t" + overallMax);

        var plot = $.plot($("#meanmag"),
            // [{data: bData, label: "Bias"},{data: WData, label: "Weights"}], {
            toPlot, {
                series: {
                    lines: {
                        show: true,
                        lineWidth: 2,
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

        var previousPoint = null;
        $("#meanmag").bind("plothover", function (event, pos, item) {
            $("#xMeanMagnitudes").text(pos.x.toFixed(0));
            $("#yMeanMagnitudes").text(pos.y.toFixed(2));


        });
    }
}

/* ---------- Activations Chart ---------- */
function renderActivationsChart(data) {

    var mean = data["activations"]["mean"];
    var stdev = data["activations"]["stdev"];
    var iter = data["activations"]["iterCount"];

    if ($("#activations").length) {
        var meanData = [];
        var meanPlus2 = [];
        var meanMinus2 = [];

        var overallMin = Number.MAX_VALUE;
        var overallMax = Number.MIN_VALUE;

        for (var i = 0; i < iter.length; i++) {
            var mp2 = mean[i] + 2*stdev[i];
            var ms2 = mean[i] - 2*stdev[i];
            overallMin = Math.min(overallMin, ms2);
            overallMax = Math.max(overallMax, mp2);
            meanData.push([iter[i], mean[i]]);
            meanPlus2.push([iter[i], mp2]);
            meanMinus2.push([iter[i], ms2]);
        }

        if(overallMin == Number.MAX_VALUE) overallMin = 0;
        if(overallMax == Number.MIN_VALUE) overallMax = 1;

        var plot = $.plot($("#activations"),
            [{data: meanData, label: "Mean"},{data: meanPlus2, label: "Mean + 2*sd"}, {data: meanMinus2, label: "Mean - 2*sd"}], {


                series: {
                    lines: {
                        show: true,
                        lineWidth: 2,
                    }
                },
                grid: {
                    hoverable: true,
                    clickable: true,
                    tickColor: "#dddddd",
                    borderWidth: 0
                },
                yaxis: {min: overallMin, max: overallMax},
                colors: ["#FA5833", "#2FABE9", "#2FABE9"]
            });

        var previousPoint = null;
        $("#activations").bind("plothover", function (event, pos, item) {
            $("#xActivations").text(pos.x.toFixed(0));
            $("#yActivations").text(pos.y.toFixed(2));
        });
    }
}

/* ---------- Learning Rate Chart ---------- */
function renderLearningRateChart(data) {

    var lrs_b = data["learningRates"]["lrs"]["b"];
    var lrs_W = data["learningRates"]["lrs"]["W"];
    var iter = data["learningRates"]["iterCounts"];

    if ($("#learningrate").length) {


        var lrs_bData = [];
        var lrs_WData = [];

        for (var i = 0; i < iter.length; i++) {
            lrs_bData.push([iter[i], lrs_b[i]]);
            lrs_WData.push([iter[i], lrs_W[i]]);
        }

        var plot = $.plot($("#learningrate"),
            [{data: lrs_bData, label: "lrs_b"},{data: lrs_WData, label: "lrs_w"}], {
                series: {
                    lines: {
                        show: true,
                        lineWidth: 2,
                    }
                },
                grid: {
                    hoverable: true,
                    clickable: true,
                    tickColor: "#dddddd",
                    borderWidth: 0
                },
                yaxis: {min: 0, max: 1},
                colors: ["#FA5833", "#2FABE9"]
            });

        var previousPoint = null;
        $("#learningrate").bind("plothover", function (event, pos, item) {
            $("#xLearningRate").text(pos.x.toFixed(0));
            $("#yLearningRate").text(pos.y.toFixed(2));
        });
    }
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
            var layerName = data["vertexNames"][1];
            var layerType = data["vertexTypes"][1];
            var inputSize = data["vertexInfo"][1]["Input size"];
            var outputSize = data["vertexInfo"][1]["Output size"];
            var nParams = data["vertexInfo"][1]["Num Parameters"];
            var activationFunction = data["vertexInfo"][1]["Activation Function"];
            var lossFunction = data["vertexInfo"][1]["Loss Function"];

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



	/* ---------- Parameters Histogram ---------- */

function renderParametersHistogram(data) {

	if($("#parametershistogram").length)
	{
		var d1 = [];
		for (var i = 0; i <= 10; i += 1)
		d1.push([i, parseInt(Math.random() * 30)]);

		var d2 = [];
		for (var i = 0; i <= 10; i += 1)
			d2.push([i, parseInt(Math.random() * 30)]);

		var d3 = [];
		for (var i = 0; i <= 10; i += 1)
			d3.push([i, parseInt(Math.random() * 30)]);

		var stack = 0, bars = true, lines = false, steps = false;

		function plotWithOptions() {
			$.plot($("#parametershistogram"), [ d1, d2, d3 ], {
				series: {
					stack: stack,
					lines: { show: lines, fill: true, steps: steps },
					bars: { show: bars, barWidth: 0.6 },
				},
				colors: ["#FA5833", "#2FABE9", "#FABB3D"]
			});
		}

		plotWithOptions();

		$(".stackControls input").click(function (e) {
			e.preventDefault();
			stack = $(this).val() == "With stacking" ? true : null;
			plotWithOptions();
		});
		$(".graphControls input").click(function (e) {
			e.preventDefault();
			bars = $(this).val().indexOf("Bars") != -1;
			lines = $(this).val().indexOf("Lines") != -1;
			steps = $(this).val().indexOf("steps") != -1;
			plotWithOptions();
		});
	}
}