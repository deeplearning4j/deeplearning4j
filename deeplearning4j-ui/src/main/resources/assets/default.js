/*

*/

var events = [];


function trackSessionHandle(event, url) {
   var sessions = events[event];
    if (sessions == undefined || sessions == null) {
        console.log("No events");
        $.notify({
            title: '<strong>No data available!</strong>',
            message: 'No sessions for ' + event + ' were registered yet...'
        },{
            type: 'warning',
            placement: {
                from: "top",
                align: "center"
            },
        });

        return false;
    }

    console.log("Number of events: " + sessions.length);
    console.log("Session[0]: " + sessions[0])
/*
    if (sessions.length == 1) {
        window.location.href = (url + "?sid=" + sessions[0]);
    }
    */
    showSessionSelector(sessions, url);
}


function showSessionSelector(sessions, url) {

    var html = "";
    for (var i = 0; i < sessions.length; i++) {
        html = html + "<a href='" + url + "?sid="+sessions[i]+"'>"+ sessions[i]+"</a>";
    }
    $("#sessionList").html(html);
    $("#sessionSelector").css("display","block");
}


/*
    This function updates event
*/
function timedCall() {
 $.ajax({
                             url:"events",
                             async: true,
                             error: function (query, status, error) {
                                 $.notify({
                                     title: '<strong>No connection!</strong>',
                                     message: 'DeepLearning4j UiServer seems to be down!'
                                 },{
                                     type: 'danger',
                                     placement: {
                                         from: "top",
                                         align: "center"
                                         },
                                 });

                                 var keys = Object.keys(events);
                                 for (var i = 0; i < keys.length; i++) {
                                        $("#"+keys[i]).fadeTo(1,0.2);
                                 }

                                 events = [];

                                 setTimeout(timedCall, 5000);
                             },
                             success: function( data ) {
                                if (data ==undefined)
                                    setTimeout(timedCall, 3000);

                                events = data;

                                var keys = Object.keys(data);
                                for (var i = 0; i < keys.length; i++) {
                                    $("#"+keys[i]).fadeTo(1,1);
                                }

                                setTimeout(timedCall, 3000);
                             }
    });
}

timedCall();