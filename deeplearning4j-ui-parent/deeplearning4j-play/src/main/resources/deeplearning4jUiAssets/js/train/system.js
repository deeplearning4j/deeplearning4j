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

            var jvmValues = data["memory"]["0"]["values"][0];
            var jvmTimes = data["memory"]["0"]["times"];

            if($("#jvm-memory-chart").length)
            {
                var jvmData = [];

                for (var i = 0; i < jvmValues.length; i++) {
                    jvmData.push([jvmTimes[i], jvmValues[i]]);
                }

                var plot = $.plot($("#jvm-memory-chart"),
                       [ { data: jvmData,
                           lines: { show: true,
                                    fill: false,
                                    lineWidth: 2
                                  },
                           shadowSize: 0
                          }, {
                            data: jvmData,
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

/* ---------- Off Heap Memory Utilization Chart ---------- */
function renderOffHeapMemoryChart() {

    $.ajax({
        url: "/train/system/data",
        async: true,
        error: function (query, status, error) {
            console.log("Error getting data: " + error);
        },
        success: function (data) {
            console.log("Keys: " + Object.keys(data));

            var offHeapValues = data["memory"]["0"]["values"][1];
            var jvmTimes = data["memory"]["0"]["times"];

            if($("#off-heap-memory-chart").length)
            {
                var jvmData = [];

                for (var i = 0; i < offHeapValues.length; i++) {
                    jvmData.push([jvmTimes[i], offHeapValues[i]]);
                }

                var plot = $.plot($("#off-heap-memory-chart"),
                       [ { data: jvmData,
                           lines: { show: true,
                                    fill: false,
                                    lineWidth: 2
                                  },
                           shadowSize: 0
                          }, {
                            data: jvmData,
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
            var jvmAvailableProcessors = data["hardware"]["0"][2][1];
            var nComputeDevices = data["hardware"]["0"][3][1];

            /* Software */
            var OS = data["software"]["0"][0][1];
            var hostName = data["software"]["0"][1][1];
            var OSArchitecture = data["software"]["0"][2][1];
            var jvmName = data["software"]["0"][3][1];
            var jvmVersion = data["software"]["0"][4][1];
            var nd4jBackend = data["software"]["0"][5][1];
            var nd4jDataType = data["software"]["0"][6][1];

            /* Memory */
            var currentBytesJVM = data["memory"]["0"]["currentBytes"][0];
            var currentBytesOffHeap = data["memory"]["0"]["currentBytes"][1];
            var isDeviceJVM = data["memory"]["0"]["isDevice"][1];
            var isDeviceOffHeap = data["memory"]["0"]["isDevice"][1];
            var maxBytesJVM = data["memory"]["0"]["maxBytes"][0];
            var maxBytesOffHeap = data["memory"]["0"]["maxBytes"][1];

            /* Inject Hardware Information */
            $("#jvmAvailableProcessors").html(jvmAvailableProcessors);
            $("#nComputeDevices").html(nComputeDevices);

            /* Inject Software Information */
            $("#OS").html(OS);
            $("#hostName").html(hostName);
            $("#OSArchitecture").html(OSArchitecture);
            $("#jvmName").html(jvmName);
            $("#jvmVersion").html(jvmVersion);
            $("#nd4jBackend").html(nd4jBackend);
            $("#nd4jDataType").html(nd4jDataType);

            /* Inject Memory Information */
            $("#currentBytesJVM").html(currentBytesJVM);
            $("#currentBytesOffHeap").html(currentBytesOffHeap);
            $("#isDeviceJVM").html(isDeviceJVM);
            $("#isDeviceOffHeap").html(isDeviceOffHeap);
            $("#maxBytesJVM").html(maxBytesJVM);
            $("#maxBytesOffHeap").html(maxBytesOffHeap);

        }
    });

}