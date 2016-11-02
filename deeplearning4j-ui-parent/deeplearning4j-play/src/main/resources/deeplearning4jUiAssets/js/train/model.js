
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
        var bData = [];
        var WData = [];

        for (var i = 0; i < iter.length; i++) {
            bData.push([iter[i], b[i]]);
            WData.push([iter[i], W[i]]);
        }

        var plot = $.plot($("#meanmag"),
            [{data: bData, label: "Bias"},{data: WData, label: "Weights"}], {
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
        $("#meanmag").bind("plothover", function (event, pos, item) {
            $("#x").text(pos.x.toFixed(0));
            $("#y").text(pos.y.toFixed(2));
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
        var stdevData = [];

        for (var i = 0; i < iter.length; i++) {
            meanData.push([iter[i], mean[i]]);
            stdevData.push([iter[i], stdev[i]]);
        }

        var plot = $.plot($("#activations"),
            [{data: meanData, label: "Mean"},{data: stdevData, label: "Standard Deviation"}], {
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
                yaxis: {min: -1, max: 1},
                colors: ["#FA5833", "#2FABE9"]
            });

        var previousPoint = null;
        $("#activations").bind("plothover", function (event, pos, item) {
            $("#x").text(pos.x.toFixed(0));
            $("#y").text(pos.y.toFixed(2));
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
            $("#x").text(pos.x.toFixed(0));
            $("#y").text(pos.y.toFixed(2));
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
            var layerName = data["layerNames"][1];
            var layerType = data["layerTypes"][1];
            var inputSize = data["layerInfo"][1]["Input size"];
            var outputSize = data["layerInfo"][1]["Output size"];
            var nParams = data["layerInfo"][1]["Num Parameters"];
            var activationFunction = data["layerInfo"][1]["Activation Function"];
            var lossFunction = data["layerInfo"][1]["Loss Function"];

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