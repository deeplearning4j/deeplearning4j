/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

var events = [];


function trackSessionHandle(event, url) {
   var sessions = events[event];
    if (sessions == undefined || sessions == null) {
        if (event == "TSNE") window.location.href = url;
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

    console.log("Number of events for [" + event + "]: " + sessions.length);

    if (sessions.length == 1) {
        window.location.href = (url + "?sid=" + sessions[0]);
    } else {
        showSessionSelector(sessions, url);
    }
}


function showSessionSelector(sessions, url) {

    var html = "";
    for (var i = 0; i < sessions.length; i++) {
        html = html + "<tr><td style='width: 100%;'><a href='" + url + "?sid="+encodeURIComponent(sessions[i])+"'>"+ sessions[i]+"</a></td></tr>";
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