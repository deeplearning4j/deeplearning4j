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
    $('#sidebar-wrapper').hide();
    // Add events
    $('#form').fileUpload({success : function(data, textStatus, jqXHR){
        document.getElementById('form').reset();
        $('#form').hide();

        $.ajax({
            url: '/word2vec/vocab',
            type: 'POST',
            data: data,
            cache: false,
            success: function(data, textStatus, jqXHR)
            {
                if(typeof data.error === 'undefined')
                {
                    // Success so call function to process the form
                    console.log('SUCCESS');
                    $(document).ready(function() {
                        $('#kform').show();
                        $('#sidebar-wrapper').show();
                        $("#instructions").hide();
                        var html = '<ul class="sidebar-nav">';
                        var keys = Object.keys(data);
                        for (var i = 0; i < keys.length; i++) {
                            html = html + '<li class="sidebar-brand"><a class ="word" href="#">' + data[keys[i]] + "</a></li>";
                        }
                        html += "</ul>";
                        document.getElementById("sidebar-wrapper").innerHTML = html;
                    });
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
    },error : function(err) {
        console.log(err);
    }});

    function renderNearestNeighbors(word, numWords) {
        $.ajax({
            url: '/word2vec/words',
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
                    var html = '<h3>Target word: <b><u>' + word + '</u></b></h3><br><h4>' + k +  ' nearest neighbors and their cosine similarity to target word: </h4> \<' +
                        'ol>';

                    for (var i = 0; i < k; i++) {
                        var sim = data[keys[i]];
                        html += '<li>' + keys[i] + '&nbsp;&nbsp;&nbsp;<span style="font-size: 10px;">[ '+ (sim.toFixed(4))+' ]</span></li>';
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
