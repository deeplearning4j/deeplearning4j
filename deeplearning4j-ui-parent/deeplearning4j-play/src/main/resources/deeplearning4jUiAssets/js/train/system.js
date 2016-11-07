selectMachine(); //Make machineID Global

var lastUpdateTimeSystem = -1;
var lastUpdateSessionSystem = "";
function renderSystemPage(firstLoad) {
    updateSessionWorkerSelect();

    if(firstLoad || !lastUpdateSessionSystem || lastUpdateSessionSystem == "" || lastUpdateSessionSystem != currSession){
        executeSystemUpdate();
    } else {
        //Check last update time first - see if data has actually changed...
        $.ajax({
            url: "/train/sessions/lastUpdate/" + currSession,
            async: true,
            error: function (query, status, error) {
                console.log("Error getting data: " + error);
            },
            success: function (data) {
                if(data > lastUpdateTimeSystem){
                    executeSystemUpdate();
                }
            }
        });
    }
}

function executeSystemUpdate(){
    $.ajax({
        url: "/train/system/data",
        async: true,
        error: function (query, status, error) {
            console.log("Error getting data: " + error);
        },
        success: function (data) {
            lastUpdateSessionSystem = currSession;
            lastUpdateTimeSystem = data["updateTimestamp"];
            renderSystemMemoryChart(data);
            renderSystemInformation(data);
            renderGPULayout(data);
           renderGpuMemoryChart(data);
        }
    });
}

function renderTabs() {
    $.ajax({
        url: "/train/system/data",
        async: true,
        error: function (query, status, error) {
            console.log("Error getting data: " + error);
        },
        success: function (data) {
            renderMultipleTabs(data);
        }
    });
}

/* ---------- System Memory Utilization Chart ---------- */
// var systemMaxLastIter = 0;
var jvmMaxLastIter = 0;
var offHeapMaxLastIter = 0;
function renderSystemMemoryChart(data) {

    var jvmCurrentFrac = data["memory"][machineID]["values"][0];
    var offHeapFrac = data["memory"][machineID]["values"][1];
    var systemChart = $("#systemMemoryChartPlot");

    // systemMaxLastIter = data["memory"][machineID]["maxBytes"][0];
    jvmMaxLastIter = data["memory"][machineID]["maxBytes"][0];
    offHeapMaxLastIter = data["memory"][machineID]["maxBytes"][1];

    if (systemChart.length) {

        var jvmValuesData = [];
        var offHeapValuesData = [];

        for (var i = 0; i < jvmCurrentFrac.length; i++) {
            jvmValuesData.push([i, 100.0 * jvmCurrentFrac[i]]);
            offHeapValuesData.push([i, 100.0 * offHeapFrac[i]]);
        }

        // console.log("JVM:" + jvmValuesData);
        // console.log("Off-Heap:" + offHeapValuesData);

        var plot = $.plot(systemChart,
            [{data: jvmValuesData, label: "JVM Memory"}, {data: offHeapValuesData, label: "Off-Heap Memory"}], {
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
                yaxis: {min: 0, max: 100.0},
                colors: ["#FA5833", "#2FABE9"]
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
            }).appendTo("#systemMemoryChartPlot").fadeIn(200);
        }

        var previousPoint = null;
        systemChart.bind("plothover", function (event, pos, item) {
            var xPos = pos.x.toFixed(0);
            $("#x").text(xPos < 0 || xPos == "-0" ? "" : xPos);
            var tempY = Math.min(100.0, pos.y);
            tempY = Math.max(tempY, 0.0);
            var asBytesJvm = formatBytes(tempY * jvmMaxLastIter / 100.0, 2);
            var asBytesOffHeap = formatBytes(tempY * offHeapMaxLastIter / 100.0, 2);
            $("#y").text(tempY.toFixed(2) + "% (" + asBytesJvm + ", " + asBytesOffHeap + ")");

            if (item) {
                if (previousPoint != item.dataIndex) {
                    previousPoint = item.dataIndex;

                    $("#tooltip").remove();
                    var x = item.datapoint[0].toFixed(0);
                    var y = Math.min(100.0, item.datapoint[1]).toFixed(2);

                    var label = item.series.label;
                    var bytes;
                    if (label.toLowerCase().startsWith("jvm")) {
                        bytes = (item.datapoint[1] * jvmMaxLastIter / 100.0).toFixed(0);
                    } else {
                        bytes = (item.datapoint[1] * offHeapMaxLastIter / 100.0).toFixed(0);
                    }

                    showTooltip(item.pageX - systemChart.offset().left, item.pageY - systemChart.offset().top,
                        item.series.label + " (" + x + ", " + y + "%; " + formatBytes(bytes, 2) + ")");
                }
            }
            else {
                $("#tooltip").remove();
                previousPoint = null;
            }
        });
    }
}

/* ---------- GPU Utilization Chart (TBD) ---------- */
var gpuMaxLastIter = {};
function renderGpuMemoryChart(data) {

    var gpuFrac = data["memory"][machineID]["values"][1];
    var gpuChart = $("#gpuMemoryChartPlot");

    var isDevice = data["memory"][machineID]["isDevice"];
    var deviceIdxs = [];
    if(isDevice ){
        for(var i=0; i<isDevice.length; i++ ){
            //if(isDevice[i] == false){     //For testing GPU chart on non-GPU system...
            if(isDevice[i] == true){
                deviceIdxs.push(i);
            }
        }
    }

    if(deviceIdxs.length == 0){
        return;
    }

    //Do have devices -> render them...
    if (gpuChart.length) {

        var toRender = [];

        for(var i=0; i<deviceIdxs.length; i++ ){
            var seriesName = data["memory"][machineID]["seriesNames"][deviceIdxs[i]];
            var thisDeviceMemFrac = data["memory"][machineID]["values"][deviceIdxs[i]];
            gpuMaxLastIter[seriesName] = data["memory"][machineID]["maxBytes"][deviceIdxs[i]];
            var xy = [];
            for( var j=0; j<thisDeviceMemFrac.length; j++ ){
                xy.push([j, 100.0 * thisDeviceMemFrac[j]])
            }
            toRender.push({data: xy, label: seriesName});
        }

        var plot = $.plot(gpuChart,
            toRender, {
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
                yaxis: {min: 0, max: 100.0},
                colors: ["#FA5833", "#2FABE9"]
            });

        function showTooltip(x, y, contents) {
            $('<div id="tooltipGpu">' + contents + '</div>').css({
                position: 'absolute',
                display: 'none',
                top: y + 8,
                left: x + 10,
                border: '1px solid #fdd',
                padding: '2px',
                'background-color': '#dfeffc',
                opacity: 0.80
            }).appendTo("#gpuMemoryChartPlot").fadeIn(200);
        }

        var previousPoint = null;
        gpuChart.bind("plothover", function (event, pos, item) {
            var xPos = pos.x.toFixed(0);
            $("#x2").text(xPos < 0 || xPos == "-0" ? "" : xPos);
            var tempY = Math.min(100.0, pos.y);
            tempY = Math.max(tempY, 0.0);
            $("#y2").text(tempY.toFixed(2) + "%");

            if (item) {
                if (previousPoint != item.dataIndex) {
                    previousPoint = item.dataIndex;

                    $("#tooltipGpu").remove();
                    var x = item.datapoint[0].toFixed(0);
                    var y = Math.min(100.0, item.datapoint[1]).toFixed(2);

                    var label = item.series.label;
                    var max = gpuMaxLastIter[item.series.label];
                    var bytes = (item.datapoint[1] * max / 100.0).toFixed(0);

                    showTooltip(item.pageX - gpuChart.offset().left, item.pageY - gpuChart.offset().top,
                        item.series.label + " (" + x + ", " + y + "%; " + formatBytes(bytes, 2) + ")");
                }
            }
            else {
                $("#tooltipGpu").remove();
                previousPoint = null;
            }
        });
    }
}

/* ---------- System Information ---------- */
function renderSystemInformation(data) {

    /* Hardware */
    var jvmAvailableProcessors = data["hardware"][machineID][2][1];
    var nComputeDevices = data["hardware"][machineID][3][1];

    /* Software */
    var OS = data["software"][machineID][0][1];
    var hostName = data["software"][machineID][1][1];
    var OSArchitecture = data["software"][machineID][2][1];
    var jvmName = data["software"][machineID][3][1];
    var jvmVersion = data["software"][machineID][4][1];
    var nd4jBackend = data["software"][machineID][5][1];
    var nd4jDataType = data["software"][machineID][6][1];

    /* Memory */
    var currentBytesJVM = data["memory"][machineID]["currentBytes"][0];
    var currentBytesOffHeap = data["memory"][machineID]["currentBytes"][1];
    var isDeviceJVM = data["memory"][machineID]["isDevice"][1];
    var isDeviceOffHeap = data["memory"][machineID]["isDevice"][1];
    var maxBytesJVM = data["memory"][machineID]["maxBytes"][0];
    var maxBytesOffHeap = data["memory"][machineID]["maxBytes"][1];

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
    $("#currentBytesJVM").html(formatBytes(currentBytesJVM, 2));
    $("#currentBytesOffHeap").html(formatBytes(currentBytesOffHeap, 2));
    $("#isDeviceJVM").html(isDeviceJVM);
    $("#isDeviceOffHeap").html(isDeviceOffHeap);
    $("#maxBytesJVM").html(formatBytes(maxBytesJVM, 2));
    $("#maxBytesOffHeap").html(formatBytes(maxBytesOffHeap, 2));

    /* Inject GPU Information (TBD) */

}

/* ---------- GPU Layout ---------- */
function renderGPULayout(data) {

    // var isDevice = data["memory"][machineID]["isDevice"][0];
    var anyDevices = false;
    var isDevice = data["memory"][machineID]["isDevice"];
    if(isDevice ){
        for(var i=0; i<isDevice.length; i++ ){
            if(isDevice[i] == true){
                anyDevices = true;
                break;
            }
        }
    }

    //anyDevices = true;    //For testing GPU charts on non-GPU system...
    if (anyDevices == true) {
        //$("#gpuTable").show();
        $("#gpuMemoryChart").show();
        $("#systemMemoryChart").attr("class", "box span6");
    }
    else {
        //$("#gpuTable").hide();
        $("#gpuMemoryChart").hide();
        $("#systemMemoryChart").attr("class", "box span12");
    }
}

/* ---------- Render System Dropdown ---------- */
function renderMultipleTabs(data) {

    var nMachinesData = data["memory"];
    var nMachines = Object.keys(nMachinesData);

    /* Generate Tabs Depending on nMachines.length*/
    for (i = 0; i < nMachines.length; i++)  {
        $('#systemTab').append("<li id=\"" + nMachines[i] + "\"><a href=\"javascript:void();\">Machine " + nMachines[i] + "</a></li>");
    }
}

/* ---------- Set Machine ID Depending on Item Clicked ---------- */
function selectMachine() {

    machineID = 0;

    $('#systemTab').on("click", "li", function () {
        machineID = $(this).attr('id');
        lastUpdateTimeSystem = -1;      //Reset last update time to force chart refresh
    });

    return machineID;
}