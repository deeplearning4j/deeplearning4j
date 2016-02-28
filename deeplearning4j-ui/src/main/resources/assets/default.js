/*

*/

var events = [];


function trackSessionHandle(event, url) {
   var sessions = events[event];
    if (sessions == undefined || sessions == null) {
        console.log("No events");
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
                                 setTimeout(timedCall, 5000);
                             },
                             success: function( data ) {
                                events = data;
                                setTimeout(timedCall, 2000);
                             }
    });
}

timedCall();