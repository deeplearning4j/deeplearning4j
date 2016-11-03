function renderSystemPage() {
    $.ajax({
        url: "/train/system/data",
        async: true,
        error: function (query, status, error) {
            console.log("Error getting data: " + error);
        },
        success: function (data) {
            renderJVMMemoryChart(data);
            renderOffHeapMemoryChart(data);
            renderSystemInformation(data);
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

selectMachine(); //Make machineID Global

/* ---------- JVM Memory Utilization Chart ---------- */
var jvmMaxLastIter = 0;
function renderJVMMemoryChart(data) {

    var jvmCurrentFrac = data["memory"][machineID]["values"][0];
    var jvmChart = $("#jvmmemorychart");

    jvmMaxLastIter = data["memory"][machineID]["maxBytes"][0];

    if (jvmChart.length) {

        var jvmValuesData = [];

        for (var i = 0; i < jvmCurrentFrac.length; i++) {
            jvmValuesData.push([i, 100.0 * jvmCurrentFrac[i]]);
        }

        var plot = $.plot(jvmChart,
            [{data: jvmValuesData, label: "JVM Memory"}], {
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
            $('<div id="tooltip">' + contents + '</div>').css( {
                position: 'absolute',
                display: 'none',
                top: y + 8,
                left: x + 10,
                border: '1px solid #fdd',
                padding: '2px',
                'background-color': '#dfeffc',
                opacity: 0.80
            }).appendTo("#jvmmemorychart").fadeIn(200);
        }

        var previousPoint = null;
        jvmChart.bind("plothover", function (event, pos, item) {
            $("#x").text(pos.x.toFixed(0));
            var tempY = Math.min(100.0,pos.y);
            tempY = Math.max(tempY, 0.0);
            var asBytes = formatBytes(tempY * jvmMaxLastIter / 100.0, 2);
            $("#y").text(tempY.toFixed(2) + " (" + asBytes + ")");

            if (item) {
                if (previousPoint != item.dataIndex) {
                    previousPoint = item.dataIndex;

                    $("#tooltip").remove();
                    var x = item.datapoint[0].toFixed(0);
                    var y = Math.min(100.0, item.datapoint[1]).toFixed(2);
                    var bytes = (item.datapoint[1] * jvmMaxLastIter / 100.0).toFixed(0);

                    showTooltip(item.pageX - jvmChart.offset().left, item.pageY - jvmChart.offset().top,
                        "(" + x + ", " + y + "%; " + formatBytes(bytes,2) + ")");
                }
            }
            else {
                $("#tooltip").remove();
                previousPoint = null;
            }
        });
    }
}

/* ---------- Off Heap Memory Utilization Chart ---------- */
var offHeapMaxLastIter = 0;
function renderOffHeapMemoryChart(data) {

    var offHeapFrac = data["memory"][machineID]["values"][1];
    var offHeapChart = $("#offheapmemorychart");

    offHeapMaxLastIter = data["memory"][machineID]["maxBytes"][1];

    if (offHeapChart.length) {

        var offHeapValuesData = [];

        for (var i = 0; i < offHeapFrac.length; i++) {
            offHeapValuesData.push([i, 100.0 * offHeapFrac[i]]);
        }

        var plot = $.plot(offHeapChart,
            [{data: offHeapValuesData, label: "Off Heap Memory"}], {
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
            $('<div id="tooltipOffHeap">' + contents + '</div>').css( {
                position: 'absolute',
                display: 'none',
                top: y + 8,
                left: x + 10,
                border: '1px solid #fdd',
                padding: '2px',
                'background-color': '#dfeffc',
                opacity: 0.80
            }).appendTo("#offheapmemorychart").fadeIn(200);
        }

        var previousPoint = null;
        offHeapChart.bind("plothover", function (event, pos, item) {
            $("#x2").text(pos.x.toFixed(0));
            var tempY = Math.min(100.0,pos.y);
            tempY = Math.max(0.0, tempY);
            var asBytes = formatBytes(tempY * offHeapMaxLastIter / 100.0, 2);
            $("#y2").text(tempY.toFixed(2) + " (" + asBytes + ")");

            if (item) {
                if (previousPoint != item.dataIndex) {
                    previousPoint = item.dataIndex;

                    $("#tooltipOffHeap").remove();
                    var x = item.datapoint[0].toFixed(0);
                    var y = Math.min(100.0, item.datapoint[1]).toFixed(2);
                    var bytes = (item.datapoint[1] * offHeapMaxLastIter / 100.0).toFixed(0);

                    showTooltip(item.pageX - offHeapChart.offset().left, item.pageY - offHeapChart.offset().top,
                        "(" + x + ", " + y + "%; " + formatBytes(bytes,2) + ")");
                }
            }
            else {
                $("#tooltipOffHeap").remove();
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
    $("#currentBytesJVM").html(formatBytes(currentBytesJVM,2));
    $("#currentBytesOffHeap").html(formatBytes(currentBytesOffHeap,2));
    $("#isDeviceJVM").html(isDeviceJVM);
    $("#isDeviceOffHeap").html(isDeviceOffHeap);
    $("#maxBytesJVM").html(formatBytes(maxBytesJVM,2));
    $("#maxBytesOffHeap").html(formatBytes(maxBytesOffHeap,2));

}


/* ---------- Render Tabs ---------- */
function renderMultipleTabs(data) {

    var nMachinesData = data["memory"];
    var nMachines = Object.keys(nMachinesData);

    /* Generate Tabs Depending on nMachines.length*/
    for (i = 0; i < nMachines.length; i++)  {
        $('#systemTab').append("<li id=\"" + nMachines[i] + "\"><a href=\"#machine" + nMachines[i] + "\">Machine" + nMachines[i] + "</a></li>");
    }

    /* Show/Hide Tabs on Click */
    $('#systemTab a:first').tab('show');
    $('#systemTab a').click(function (e) {
      e.preventDefault();
      $(this).tab('show');
    });

}

/* Set Machine ID Depending on Tab Clicked */
function selectMachine() {

    machineID = 0;

    $('#systemTab').on("click", "li", function() {
        machineID = $(this).attr('id');
     });

    return machineID;
}

/* ---------- Language Dropdown ---------- */

	$('.dropmenu').click(function(e){
		e.preventDefault();
		$(this).parent().find('ul').slideToggle();
	});