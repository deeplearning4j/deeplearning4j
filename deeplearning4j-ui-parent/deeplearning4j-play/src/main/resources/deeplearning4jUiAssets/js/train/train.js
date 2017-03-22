
function languageSelect(languageCode, redirect){
    //language code: iso639 code

    $.ajax({
        url: "/setlang/" + languageCode,
        async: true,
        error: function (query, status, error) {
            console.log("Error getting data: " + error);
        },
        success: function (data) {
            window.location.replace('/train/' + redirect)
        }
    });
}

var currSession = "";
var currWorkerIdx = 0;
var prevNumWorkers = 0;
function updateSessionWorkerSelect(){

    $.ajax({
        url: "/train/sessions/current",
        async: true,
        error: function (query, status, error) {
            console.log("Error getting data: " + error);
        },
        success: function (data) {
            currSession = data;

            $.ajax({
                url: "/train/sessions/info",
                async: true,
                error: function (query, status, error) {
                    console.log("Error getting data: " + error);
                },
                success: function (data) {
                    var keys = Object.keys(data);
                    if(keys.length > 1) {   //only show session selector if there are multiple sessions

                        var elem = $("#sessionSelect");
                        elem.empty();

                        var currSelectedIdx = 0;
                        for (var i = 0; i < keys.length; i++) {
                            if(keys[i] == currSession){
                                currSelectedIdx = i;
                            }
                            elem.append("<option value='" + keys[i] + "'>" + keys[i] + "</option>");
                        }

                        $("#sessionSelect option[value='" + keys[currSelectedIdx] +"']").attr("selected", "selected");
                        $("#sessionSelectDiv").show();
                    } else {
                        $("#sessionSelectDiv").hide();
                    }

                    //Set up worker selection...
                    if(data[currSession]){
                        var numWorkers = data[currSession]["numWorkers"];
                        var workers = data[currSession]["workers"];

                        var elem = $("#workerSelect");
                        elem.empty();

                        if(numWorkers > 1){
//                        if(numWorkers >= 0){    //For testing
                            for(var i=0; i<workers.length; i++){
                                elem.append("<option value='" + i + "'>" + workers[i] + "</option>");
                            }

                            $("#workerSelect option[value='" + currWorkerIdx +"']").attr("selected", "selected");
                            $("#workerSelectDiv").show();
                        } else {
                            $("#workerSelectDiv").hide();
                        }

                        // if workers change then reset
                        if(prevNumWorkers != numWorkers) {
                            if(numWorkers==0) {
                                $("#workerSelect").val("0");
                                selectNewWorker();
                            }
                            else selectNewWorker();
                        }
                    }
                }
            });
        }
    });
}

function selectNewSession(){
    var selector = $("#sessionSelect");
    var currSelected = selector.val();

    if(currSelected){
        $.ajax({
            url: "/train/sessions/set/" + currSelected,
            async: true,
            error: function (query, status, error) {
                console.log("Error setting session: " + error);
            },
            success: function (data) {
            }
        });
    }
}


function selectNewWorker(){
    var selector = $("#workerSelect");
    var currSelected = selector.val();

    if(currSelected){
        $.ajax({
            url: "/train/workers/setByIdx/" + currSelected,
            async: true,
            error: function (query, status, error) {
                console.log("Error setting session: " + error);
            },
            success: function (data) {
                currWorkerIdx = currSelected;
            }
        });
    }
}


function formatBytes(bytes, precision){
    var index = 0;
    var newValue = bytes;
    while(newValue > 1024){
        newValue = newValue/1024;
        index++;
    }

    var unit = "";
    switch (index){
        case 0:
            unit = "B";
            break;
        case 1:
            unit = "kB";
            break;
        case 2:
            unit = "MB";
            break;
        case 3:
            unit = "GB";
            break;
        case 4:
            unit = "TB";
            break;
    }

    return newValue.toFixed(precision) + " " + unit;
}

/* ---------- Language Dropdown ---------- */

	$('.dropmenu').click(function(e){
		e.preventDefault();
		$(this).parent().find('ul').slideToggle();
	});