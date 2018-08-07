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

function buildSessionSelector(event) {
  $.ajax({
                              url:"/sessions?event=" + event,
                              async: true,
                              error: function (query, status, error) {
                                  /*$.notify({
                                      title: '<strong>No connection!</strong>',
                                      message: 'DeepLearning4j UiServer seems to be down!'
                                  },{
                                      type: 'danger',
                                      placement: {
                                          from: "top",
                                          align: "center"
                                          },
                                  });
                                  setTimeout(buildSessionSelector(event), 2000);
                                  */
                              },
                              success: function( data ) {
                                 if (data == undefined || data.length == 0) {
                                    $("#sessionSelector").append("<option value='0' selected>No sessions available</option>");
                                    return;
                                 }



                                 for (var i = 0; i < data.length; i++ ) {
                                    $("#sessionSelector").append("<option value='"+ data[0]+"'>ID: "+data[0]+"</option>");
                                 }
                              }
     });
}

    function getParameterByName(name) {
        url = window.location.href;
        name = name.replace(/[\[\]]/g, "\\$&");
        var regex = new RegExp("[?&]" + name + "(=([^&#]*)|&|#|$)"),
            results = regex.exec(url);
        if (!results) return null;
        if (!results[2]) return '';
        return decodeURIComponent(results[2].replace(/\+/g, " "));
    }