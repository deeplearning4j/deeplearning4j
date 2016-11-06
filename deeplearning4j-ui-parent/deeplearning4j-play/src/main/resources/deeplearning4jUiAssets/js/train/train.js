
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
function updateSessionSelect(){

    $.ajax({
        url: "/train/sessions/current",
        async: true,
        error: function (query, status, error) {
            console.log("Error getting data: " + error);
        },
        success: function (data) {
            currSession = data;

            $.ajax({
                url: "/train/sessions/all",
                async: true,
                error: function (query, status, error) {
                    console.log("Error getting data: " + error);
                },
                success: function (data) {
                    if(data.length > 1) {   //only show session selector if there are multiple sessions

                        var elem = $("#sessionSelect");
                        elem.empty();

                        var currSelectedIdx = 0;
                        for (var i = 0; i < data.length; i++) {
                            if(data[i] == currSession){
                                currSelectedIdx = i;
                            }
                            elem.append("<option value='" + data[i] + "'>" + data[i] + "</option>");
                        }

                        $("#sessionSelect option[value='" + data[currSelectedIdx] +"']").attr("selected", "selected");
                        $("#sessionSelectDiv").show();
                    } else {
                        $("#sessionSelectDiv").hide();
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