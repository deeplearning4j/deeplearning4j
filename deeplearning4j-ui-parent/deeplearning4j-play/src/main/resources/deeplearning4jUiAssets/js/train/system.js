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
function renderJVMMemoryChart(data) {

    var jvmValues = data["memory"][machineID]["values"][0];
    var maxValue = Math.max.apply(Math, jvmValues) + 1;
    var jvmChart = $("#jvmmemorychart");

    if (jvmChart.length) {

        var jvmValuesData = [];

        for (var i = 0; i < jvmValues.length; i++) {
            jvmValuesData.push([i, jvmValues[i]]);
        }

        var plot = $.plot($("#jvmmemorychart"),
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
                yaxis: {min: 0, max: maxValue},
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
            $("#y").text(pos.y.toFixed(2));

            if (item) {
                if (previousPoint != item.dataIndex) {
                    previousPoint = item.dataIndex;

                    $("#tooltip").remove();
                    var x = item.datapoint[0].toFixed(0),
                        y = item.datapoint[1].toFixed(5);

                    showTooltip(item.pageX - jvmChart.offset().left, item.pageY - jvmChart.offset().top,
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

/* ---------- Off Heap Memory Utilization Chart ---------- */
function renderOffHeapMemoryChart(data) {

    var offHeapValues = data["memory"][machineID]["values"][1];
    var maxValue = Math.max.apply(Math, offHeapValues);
    var offHeapChart = $("#offheapmemorychart");

    if (offHeapChart.length) {

        var offHeapValuesData = [];

        for (var i = 0; i < offHeapValues.length; i++) {
            offHeapValuesData.push([i, offHeapValues[i]]);
        }

        var plot = $.plot($("#offheapmemorychart"),
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
                yaxis: {min: 0, max: maxValue},
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
            }).appendTo("#offheapmemorychart").fadeIn(200);
        }

        var previousPoint = null;
        offHeapChart.bind("plothover", function (event, pos, item) {
            $("#x2").text(pos.x.toFixed(0));
            $("#y2").text(pos.y.toFixed(5));

            if (item) {
                if (previousPoint != item.dataIndex) {
                    previousPoint = item.dataIndex;

                    $("#tooltip").remove();
                    var x = item.datapoint[0].toFixed(0),
                        y = item.datapoint[1].toFixed(5);

                    showTooltip(item.pageX - offHeapChart.offset().left, item.pageY - offHeapChart.offset().top,
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
    $("#currentBytesJVM").html(currentBytesJVM);
    $("#currentBytesOffHeap").html(currentBytesOffHeap);
    $("#isDeviceJVM").html(isDeviceJVM);
    $("#isDeviceOffHeap").html(isDeviceOffHeap);
    $("#maxBytesJVM").html(maxBytesJVM);
    $("#maxBytesOffHeap").html(maxBytesOffHeap);

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