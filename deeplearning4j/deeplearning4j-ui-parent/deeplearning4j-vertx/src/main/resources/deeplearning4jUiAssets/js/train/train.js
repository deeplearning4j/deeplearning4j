
/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

function languageSelect(languageCode, redirect){
    //language code: iso639 code

    var languageSettingUrl = multiSession ? "/setlang/" + currSession + "/" + languageCode : "/setlang/" + languageCode;
    $.ajax({
        url: languageSettingUrl,
        async: true,
        error: function (query, status, error) {
            console.log("Error getting data: " + error);
        },
        success: function (data) {
            redirectUrl = multiSession ? '/train/' + currSession + "/" + redirect : '/train/' + redirect;
            window.location.replace(redirectUrl);
        }
    });
}

var multiSession = null;
var currSession = "";
var currWorkerIdx = 0;
var prevNumWorkers = 0;

function doUpdateSessionWorkerSelect() {
    var sessionInfoUrl = multiSession ? "/train/" + currSession + "/info" : "/train/sessions/info";
    $.ajax({
        url: sessionInfoUrl,
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

function getSessionIdFromUrl() {
    // path is like /train/:sessionId/overview
    var sessionIdRegexp = /\/train\/([^\/]+)\/(.*)/g;
    var match = sessionIdRegexp.exec(window.location.pathname)
    return match[1];
}

function getCurrSession(callback) {
    if (multiSession) {
        if (currSession == "") {
            // get only once
            currSession = getSessionIdFromUrl();
        }
        callback();
    } else {
        $.ajax({
            url: "/train/sessions/current",
            async: true,
            error: function (query, status, error) {
                console.log("Error getting data: " + error);
            },
            success: function (data) {
                currSession = data;
                callback();
            }
        });
    }
}


function getSessionSettings(callback) {
    // load only once
    if (multiSession != null) {
        getCurrSession(callback);
    } else {
        $.ajax({
            url: "/train/multisession",
            async: true,
            error: function (query, status, error) {
                console.log("Error getting data: " + error);
            },
            success: function (data) {
                multiSession = data == "true";
                getCurrSession(callback);
            }
        });
    }

}

function updateSessionWorkerSelect(){
    getSessionSettings(doUpdateSessionWorkerSelect);
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