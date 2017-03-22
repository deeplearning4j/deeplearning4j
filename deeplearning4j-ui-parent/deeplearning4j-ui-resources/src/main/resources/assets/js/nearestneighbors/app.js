/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

/**
 * Created by sonali on 4/29/15.
 */



$(document).ready(function() {
    $('#kform').hide();
    // Add events
    $('#form').fileUpload({success : function(data, textStatus, jqXHR){
        document.getElementById('form').reset();
        $('#form').hide();
        loadVocab();
    },error : function(err) {
        console.log(err);
    }});

    $('#urlsubmit').click(function() {
        var val = $('#urlval').val();
        $.ajax({
            url: '/nearestneighbors/update',
            type: 'POST',
            dataType: 'json',
            contentType : 'application/json',
            data: JSON.stringify({"url" : val}),
            cache: false,
            success: function(data, textStatus, jqXHR) {
                loadVocab();

            },
            error: function(jqXHR, textStatus, errorThrown) {
                // Handle errors here
                console.log('ERRORS: ' + textStatus);
            },
            complete: function() {
            }
        });
    });



    function loadVocab() {
        $.ajax({
            url: '/nearestneighbors/vocab',
            type: 'POST',
            data: JSON.stringify({}),
            cache: false,
            success: function(data, textStatus, jqXHR)
            {
                if(typeof data.error === 'undefined')
                {
                    // Success so call function to process the form
                    console.log('SUCCESS');
                    $('#kform').show();
                    $('#form').hide();
                    $('#url').hide();
                    var html = '<ul class="sidebar-nav">';
                    var keys = Object.keys(data);
                    for (var i = 0; i < keys.length; i++) {
                        html = html + '<li class="sidebar-brand"><a class ="word" href="#">' + data[keys[i]] + "</a></li>";
                    }
                    html += "</ul>";
                    document.getElementById("sidebar-wrapper").innerHTML = html;

                    //on click of any word, render the k nearest neighbors
                    $('.word').on('click', function(evt) {
                        var data = $(this).html();
                        var kval = $('#k').val();
                        renderNearestNeighbors(data, kval);
                    })
                }
                else
                {
                    // Handle errors here
                    console.log('ERRORS: ' + data.error);
                }
            },
            error: function(jqXHR, textStatus, errorThrown)
            {
                // Handle errors here
                console.log('ERRORS: ' + textStatus);
            },
            complete: function()
            {
                // STOP LOADING SPINNER
            }
        });
    }


    function renderNearestNeighbors(word, numWords) {
        $.ajax({
            url: '/nearestneighbors/words',
            type: 'POST',
            contentType : 'application/json',
            data: JSON.stringify({word: word, numWords: numWords}),
            cache: false,
            success: function(data, textStatus, jqXHR)
            {
                if(typeof data.error === 'undefined')
                {
                    // Success so call function to process the form
                    console.log('SUCCESS NN');
                    var keys = Object.keys(data);
                    var k = keys.length;
                    var html = '<h3>Your selected word: <b>' + word + '</b></h3><br><h4>The following are the ' + k +  ' nearest neighbors: </h4> \<' +
                        'ol>';

                    for (var i = 0; i < k; i++) {
                        html += '<li>' + keys[i] + '</li>';
                    }
                    html += '</ol>';
                    $('#neighbors').html(html);
                    //on click of any LI element, call renderNearestNeighbors (pass in word and 5)
                }
                else
                {
                    // Handle errors here
                    console.log('ERRORS: ' + data.error);
                }
            },
            error: function(jqXHR, textStatus, errorThrown)
            {
                // Handle errors here
                console.log('ERRORS: ' + textStatus);
            },
            complete: function()
            {
                // STOP LOADING SPINNER
            }
        });
    }
});
