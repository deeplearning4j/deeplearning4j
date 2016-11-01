/* ---------- JVM Memory Utilization Chart ---------- */
function renderJVMMemoryChart() {

    $.ajax({
        url: "/train/system/data",
        async: true,
        error: function (query, status, error) {
            console.log("Error getting data: " + error);
        },
        success: function (data) {
            console.log("Keys: " + Object.keys(data));

            var jvmValues = data["memory"].0.values[0];
            var jvmTimes = data["memory"].0.times;

            if($("#jvm-memory-chart").length)
            {
                var jvmData = [];

                for (var i = 0; i < jvmValues.length; i++) {
                    jvmData.push([jvmTimes[i], jvmValues[i]]);
                }

                var plot = $.plot($("#jvm-memory-chart"),
                       [ { data: jvmTimes,
                           label: "Times",
                           lines: { show: true,
                                    fill: false,
                                    lineWidth: 2
                                  },
                           shadowSize: 0
                          }, {
                            data: jvmValues,
                            bars: { show: true,
                                    fill: false,
                                    barWidth: 0.1,
                                    align: "center",
                                    lineWidth: 5}
                          }
                        ], {

                       grid: { hoverable: true,
                               clickable: true,
                               tickColor: "rgba(255,255,255,0.05)",
                               borderWidth: 0
                             },
                      legend: {show: false},
                      colors: ["rgba(255,255,255,0.8)", "rgba(255,255,255,0.6)", "rgba(255,255,255,0.4)", "rgba(255,255,255,0.2)"],
                        xaxis: {ticks:15, tickDecimals: 0, color: "rgba(255,255,255,0.8)" },
                        yaxis: {ticks:5, tickDecimals: 0, color: "rgba(255,255,255,0.8)" },
                        });
            }
        }
    });

}

/* ---------- Off-Heap Memory Utilization Chart ---------- */
/*function renderScoreChart() {

    $.ajax({
        url: "/train/system/data",
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

}*/

/* ---------- System Information ---------- */
function renderSystemInformation() {

    $.ajax({
        url: "/train/system/data",
        async: true,
        error: function (query, status, error) {
            console.log("Error getting data: " + error);
        },
        success: function (data) {
            console.log("Keys: " + Object.keys(data));

            /* Hardware */
            var jvmAvailableProcessors = data["hardware"].0[2][1];
            var nComputeDevices = data["hardware"].0[3][1];

//            /* Software */
//            var OS = data["software"][0][1];
//            var hostName = data["software"][1][1];
//            var OSArchitecture = data["software"][2][1];
//            var jvmName = data["software"][3][1];
//            var jvmVersion = data["software"][4][1];
//            var nd4jBackend = data["software"][5][1];
//            var nd4jDataType = data["software"][6 ][1];
//
//            /* Memory */
//            var currentBytesJVM = data["memory"]["currentBytes"][1];
//            var currentBytesOffHeap = data["memory"][0][2];
//            var isDeviceJVM = data["memory"][0][1];
//            var isDeviceOffHeap = data["memory"][0][1];
//            var maxBytesJVM = data["memory"][0][1];
//            var maxBytesOffHeap = data["memory"][0][1];

            /* Inject Hardware Information */
//            $("#jvmMaxMemory").html(jvmMaxMemory);
//            $("#offHeapMaxMemory").html(offHeapMaxMemory);
            $("#jvmAvailableProcessors").html(jvmAvailableProcessors);
            $("#nComputeDevices").html(nComputeDevices);

//            /* Inject Software Information */
//            $("#OS").html(OS);
//            $("#hostName").html(hostName);
//            $("#OSArchitecture").html(OSArchitecture);
//            $("#jvmName").html(jvmName);
//            $("#jvmVersion").html(jvmVersion);
//            $("#nd4jBackend").html(nd4jBackend);
//            $("#nd4jDataType").html(nd4jDataType);
//
//            /* Inject Memory Information */
//            $("#currentBytesJVM").html(currentBytesJVM);
//            $("#currentBytesOffHeap").html(jvmMaxMemoryOffHeap);
//            $("#isDeviceJVM").html(isDeviceJVM);
//            $("#isDeviceOffHeap").html(isDeviceOffHeap);
//            $("#maxBytesJVM").html(maxBytesJVM);
//            $("#maxBytesOffHeap").html(maxBytesOffHeap);
        }
    });

}