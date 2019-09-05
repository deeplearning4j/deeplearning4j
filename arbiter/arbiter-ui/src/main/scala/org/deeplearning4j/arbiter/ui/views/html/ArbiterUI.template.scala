/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.deeplearning4j.arbiter.ui.views.html

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object ArbiterUI_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class ArbiterUI extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template0[play.twirl.api.HtmlFormat.Appendable] {

  /**/
  def apply():play.twirl.api.HtmlFormat.Appendable = {
    _display_ {
      {


Seq[Any](format.raw/*1.1*/("""<!--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ~ Copyright (c) 2015-2018 Skymind, Inc.
  ~
  ~ This program and the accompanying materials are made available under the
  ~ terms of the Apache License, Version 2.0 which is available at
  ~ https://www.apache.org/licenses/LICENSE-2.0.
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
  ~ WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
  ~ License for the specific language governing permissions and limitations
  ~ under the License.
  ~
  ~ SPDX-License-Identifier: Apache-2.0
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~-->

<html>
    <head>
        <style type="text/css">
        /* Color and style reference.
            To change: do find + replace on comment + color

            heading background:         headingbgcol        #063E53            //Old candidates: #3B5998
            heading text color:         headingtextcolor    white
        */

        .hd """),format.raw/*27.13*/("""{"""),format.raw/*27.14*/("""
            """),format.raw/*28.13*/("""background-color: #000000;
            height: 41px;
            font-size: 20px;
            color: #FFFFFF;
            font-family: "Open Sans", sans-serif;
            font-weight: 200;
        """),format.raw/*34.9*/("""}"""),format.raw/*34.10*/("""

        """),format.raw/*36.9*/("""html, body """),format.raw/*36.20*/("""{"""),format.raw/*36.21*/("""
            """),format.raw/*37.13*/("""width: 100%;
            height: 100%;
            padding: 0;
        """),format.raw/*40.9*/("""}"""),format.raw/*40.10*/("""

        """),format.raw/*42.9*/(""".bgcolor """),format.raw/*42.18*/("""{"""),format.raw/*42.19*/("""
            """),format.raw/*43.13*/("""background-color: #EFEFEF;
        """),format.raw/*44.9*/("""}"""),format.raw/*44.10*/("""

        """),format.raw/*46.9*/("""h1 """),format.raw/*46.12*/("""{"""),format.raw/*46.13*/("""
            """),format.raw/*47.13*/("""font-family: "Open Sans", sans-serif;
            font-size: 28px;
            font-style: bold;
            font-variant: normal;
            font-weight: 500;
            line-height: 26.4px;
        """),format.raw/*53.9*/("""}"""),format.raw/*53.10*/("""

        """),format.raw/*55.9*/("""h3 """),format.raw/*55.12*/("""{"""),format.raw/*55.13*/("""
            """),format.raw/*56.13*/("""font-family: "Open Sans", sans-serif;
            font-size: 16px;
            font-style: normal;
            font-variant: normal;
            font-weight: 500;
            line-height: 26.4px;
        """),format.raw/*62.9*/("""}"""),format.raw/*62.10*/("""

        """),format.raw/*64.9*/("""table """),format.raw/*64.15*/("""{"""),format.raw/*64.16*/("""
            """),format.raw/*65.13*/("""font-family: "Open Sans", sans-serif;
            font-size: 14px;
        """),format.raw/*67.9*/("""}"""),format.raw/*67.10*/("""

        """),format.raw/*69.9*/("""table.resultsTable """),format.raw/*69.28*/("""{"""),format.raw/*69.29*/("""
            """),format.raw/*70.13*/("""border-collapse:collapse;
            background-color: white;
            /*border-collapse: collapse;*/
            padding: 15px;
        """),format.raw/*74.9*/("""}"""),format.raw/*74.10*/("""

        """),format.raw/*76.9*/("""table.resultsTable td, table.resultsTable tr, table.resultsTable th """),format.raw/*76.77*/("""{"""),format.raw/*76.78*/("""
            """),format.raw/*77.13*/("""border:solid black 1px;
            white-space: pre;   /* assume text is preprocessed for formatting */
        """),format.raw/*79.9*/("""}"""),format.raw/*79.10*/("""

        """),format.raw/*81.9*/("""table.resultsTable th """),format.raw/*81.31*/("""{"""),format.raw/*81.32*/("""
            """),format.raw/*82.13*/("""background-color: /*headingbgcol*/#063E53;
            color: white;
            padding-left: 4px;
            padding-right: 4px;
        """),format.raw/*86.9*/("""}"""),format.raw/*86.10*/("""

        """),format.raw/*88.9*/("""table.resultsTable td """),format.raw/*88.31*/("""{"""),format.raw/*88.32*/("""
            """),format.raw/*89.13*/("""/*background-color: white;*/
            padding-left: 4px;
            padding-right: 4px;
        """),format.raw/*92.9*/("""}"""),format.raw/*92.10*/("""

        """),format.raw/*94.9*/("""/* Properties for table cells in the tables generated using the RenderableComponent mechanism */
        .renderableComponentTable """),format.raw/*95.35*/("""{"""),format.raw/*95.36*/("""
            """),format.raw/*96.13*/("""/*table-layout:fixed; */    /*Avoids scrollbar, but makes fixed width for all columns :( */
            width: 100%
        """),format.raw/*98.9*/("""}"""),format.raw/*98.10*/("""
        """),format.raw/*99.9*/(""".renderableComponentTable td """),format.raw/*99.38*/("""{"""),format.raw/*99.39*/("""
            """),format.raw/*100.13*/("""padding-left: 4px;
            padding-right: 4px;
            white-space: pre;   /* assume text is pre-processed (important for line breaks etc)*/
            word-wrap:break-word;
            vertical-align: top;
        """),format.raw/*105.9*/("""}"""),format.raw/*105.10*/("""

        """),format.raw/*107.9*/("""/** CSS for result table rows */
        .resultTableRow """),format.raw/*108.25*/("""{"""),format.raw/*108.26*/("""
            """),format.raw/*109.13*/("""background-color: #FFFFFF;
            cursor: pointer;
        """),format.raw/*111.9*/("""}"""),format.raw/*111.10*/("""

        """),format.raw/*113.9*/("""/** CSS for result table CONTENT rows (i.e., only visible when expanded) */
        .resultTableRowSelected """),format.raw/*114.33*/("""{"""),format.raw/*114.34*/("""
            """),format.raw/*115.13*/("""background-color: rgba(0, 157, 255, 0.16);
        """),format.raw/*116.9*/("""}"""),format.raw/*116.10*/("""

        """),format.raw/*118.9*/(""".resultsHeadingDiv """),format.raw/*118.28*/("""{"""),format.raw/*118.29*/("""
            """),format.raw/*119.13*/("""background-color: /*headingbgcol*/#063E53;
            color: white;
            font-family: "Open Sans", sans-serif;
            font-size: 16px;
            font-style: bold;
            font-variant: normal;
            font-weight: 500;
            line-height: 26.4px;
            cursor: default;
            padding-top: 8px;
            padding-bottom: 8px;
            padding-left: 45px;
            padding-right: 45px;
            border-style: solid;
            border-width: 1px;
            border-color: #AAAAAA;
        """),format.raw/*135.9*/("""}"""),format.raw/*135.10*/("""

        """),format.raw/*137.9*/("""div.outerelements """),format.raw/*137.27*/("""{"""),format.raw/*137.28*/("""
            """),format.raw/*138.13*/("""padding-bottom: 30px;
        """),format.raw/*139.9*/("""}"""),format.raw/*139.10*/("""

        """),format.raw/*141.9*/("""#accordion, #accordion2 """),format.raw/*141.33*/("""{"""),format.raw/*141.34*/("""
            """),format.raw/*142.13*/("""padding-bottom: 20px;
        """),format.raw/*143.9*/("""}"""),format.raw/*143.10*/("""

        """),format.raw/*145.9*/("""#accordion .ui-accordion-header, #accordion2 .ui-accordion-header, #accordion3 .ui-accordion-header """),format.raw/*145.109*/("""{"""),format.raw/*145.110*/("""
            """),format.raw/*146.13*/("""background-color: /*headingbgcolor*/#063E53;      /*Color when collapsed*/
            color: /*headingtextcolor*/white;
            font-family: "Open Sans", sans-serif;
            font-size: 16px;
            font-style: bold;
            font-variant: normal;
            margin: 0px;
            background-image: none;     /* Necessary, otherwise color changes don't make a difference */
        """),format.raw/*154.9*/("""}"""),format.raw/*154.10*/("""

        """),format.raw/*156.9*/("""#accordion .ui-accordion-content """),format.raw/*156.42*/("""{"""),format.raw/*156.43*/("""
            """),format.raw/*157.13*/("""width: 100%;
            background-color: white;    /*background color of accordian content (elements in front may have different color */
            color: black;  /* text etc color */
            font-size: 10pt;
            line-height: 16pt;
            overflow:visible !important;
        """),format.raw/*163.9*/("""}"""),format.raw/*163.10*/("""

        """),format.raw/*165.9*/("""/** Line charts */
        path """),format.raw/*166.14*/("""{"""),format.raw/*166.15*/("""
            """),format.raw/*167.13*/("""stroke: steelblue;
            stroke-width: 2;
            fill: none;
        """),format.raw/*170.9*/("""}"""),format.raw/*170.10*/("""
        """),format.raw/*171.9*/(""".axis path, .axis line """),format.raw/*171.32*/("""{"""),format.raw/*171.33*/("""
            """),format.raw/*172.13*/("""fill: none;
            stroke: #000;
            shape-rendering: crispEdges;
        """),format.raw/*175.9*/("""}"""),format.raw/*175.10*/("""
        """),format.raw/*176.9*/(""".tick line """),format.raw/*176.20*/("""{"""),format.raw/*176.21*/("""
            """),format.raw/*177.13*/("""opacity: 0.2;
            shape-rendering: crispEdges;
        """),format.raw/*179.9*/("""}"""),format.raw/*179.10*/("""

        """),format.raw/*181.9*/("""</style>
        <title>DL4J - Arbiter UI</title>
    </head>
    <body class="bgcolor">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link id="bootstrap-style" href="/assets/webjars/bootstrap/4.3.1/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="/assets/webjars/jquery/2.2.0/jquery.min.js"></script>
        <link href="/assets/webjars/jquery-ui/1.10.2/themes/base/jquery-ui.css" rel="stylesheet">
        <script src="/assets/webjars/jquery-ui/1.10.2/ui/minified/jquery-ui.min.js"></script>
        <script src="/assets/webjars/d3js/3.3.5/d3.min.js" charset="utf-8"></script>
        <script src="/assets/webjars/bootstrap/4.3.1/dist/js/bootstrap.min.js"></script>
        <script src="/assets/dl4j-ui.js"></script>

    <script>
    //Store last update times:
    var lastStatusUpdateTime = -1;
    var lastSettingsUpdateTime = -1;
    var lastResultsUpdateTime = -1;

    var resultTableSortIndex = 0;
    var resultTableSortOrder = "ascending";
    var resultsTableContent;

    var selectedCandidateIdx = null;

    //Set basic interval function to do updates
    setInterval(doUpdate,5000);    //Loop every 5 seconds


    function doUpdate()"""),format.raw/*210.24*/("""{"""),format.raw/*210.25*/("""
        """),format.raw/*211.9*/("""//Get the update status, and do something with it:
        $.get("/arbiter/lastUpdate",function(data)"""),format.raw/*212.51*/("""{"""),format.raw/*212.52*/("""
            """),format.raw/*213.13*/("""//Encoding: matches names in UpdateStatus class
            var jsonObj = JSON.parse(JSON.stringify(data));
            var statusTime = jsonObj['statusUpdateTime'];
            var settingsTime = jsonObj['settingsUpdateTime'];
            var resultsTime = jsonObj['resultsUpdateTime'];
            //console.log("Last update times: " + statusTime + ", " + settingsTime + ", " + resultsTime);

            //Update available sessions:
            var currSession;
            $.get("/arbiter/sessions/current", function(data)"""),format.raw/*222.62*/("""{"""),format.raw/*222.63*/("""
                """),format.raw/*223.17*/("""currSession = data; //JSON.stringify(data);
                console.log("Current: " + currSession);
            """),format.raw/*225.13*/("""}"""),format.raw/*225.14*/(""");

            $.get("/arbiter/sessions/all", function(data)"""),format.raw/*227.58*/("""{"""),format.raw/*227.59*/("""
                """),format.raw/*228.17*/("""var keys = data;    // JSON.stringify(data);

                if(keys.length > 1)"""),format.raw/*230.36*/("""{"""),format.raw/*230.37*/("""
                    """),format.raw/*231.21*/("""$("#sessionSelectDiv").show();

                    var elem = $("#sessionSelect");
                    elem.empty();

                    var currSelectedIdx = 0;
                    for (var i = 0; i < keys.length; i++) """),format.raw/*237.59*/("""{"""),format.raw/*237.60*/("""
                        """),format.raw/*238.25*/("""if(keys[i] == currSession)"""),format.raw/*238.51*/("""{"""),format.raw/*238.52*/("""
                            """),format.raw/*239.29*/("""currSelectedIdx = i;
                        """),format.raw/*240.25*/("""}"""),format.raw/*240.26*/("""
                        """),format.raw/*241.25*/("""elem.append("<option value='" + keys[i] + "'>" + keys[i] + "</option>");
                    """),format.raw/*242.21*/("""}"""),format.raw/*242.22*/("""

                    """),format.raw/*244.21*/("""$("#sessionSelect option[value='" + keys[currSelectedIdx] +"']").attr("selected", "selected");
                    $("#sessionSelectDiv").show();
                """),format.raw/*246.17*/("""}"""),format.raw/*246.18*/("""
"""),format.raw/*247.1*/("""//                console.log("Got sessions: " + keys + ", current: " + currSession);
            """),format.raw/*248.13*/("""}"""),format.raw/*248.14*/(""");


            //Check last update times for each part of document, and update as necessary
            //First section: summary status
            if(lastStatusUpdateTime != statusTime)"""),format.raw/*253.51*/("""{"""),format.raw/*253.52*/("""
                """),format.raw/*254.17*/("""//Get JSON: address set by SummaryStatusResource
                $.get("/arbiter/summary",function(data)"""),format.raw/*255.56*/("""{"""),format.raw/*255.57*/("""
                    """),format.raw/*256.21*/("""var summaryStatusDiv = $('#statusdiv');
                    summaryStatusDiv.html('');

                    var str = JSON.stringify(data);
                    var component = Component.getComponent(str);
                    component.render(summaryStatusDiv);
                """),format.raw/*262.17*/("""}"""),format.raw/*262.18*/(""");

                lastStatusUpdateTime = statusTime;
            """),format.raw/*265.13*/("""}"""),format.raw/*265.14*/("""

            """),format.raw/*267.13*/("""//Second section: Optimization settings
            if(lastSettingsUpdateTime != settingsTime)"""),format.raw/*268.55*/("""{"""),format.raw/*268.56*/("""
                """),format.raw/*269.17*/("""//Get JSON for components
                $.get("/arbiter/config",function(data)"""),format.raw/*270.55*/("""{"""),format.raw/*270.56*/("""
                    """),format.raw/*271.21*/("""var str = JSON.stringify(data);

                    var configDiv = $('#settingsdiv');
                    configDiv.html('');

                    var component = Component.getComponent(str);
                    component.render(configDiv);
                """),format.raw/*278.17*/("""}"""),format.raw/*278.18*/(""");

                lastSettingsUpdateTime = settingsTime;
            """),format.raw/*281.13*/("""}"""),format.raw/*281.14*/("""

            """),format.raw/*283.13*/("""//Third section: Summary results table (summary info for each candidate)
            if(lastResultsUpdateTime != resultsTime)"""),format.raw/*284.53*/("""{"""),format.raw/*284.54*/("""

                """),format.raw/*286.17*/("""//Get JSON; address set by SummaryResultsResource
                $.get("/arbiter/results",function(data)"""),format.raw/*287.56*/("""{"""),format.raw/*287.57*/("""
                    """),format.raw/*288.21*/("""//Expect an array of CandidateInfo type objects here
                    resultsTableContent = data;
                    drawResultTable();
                """),format.raw/*291.17*/("""}"""),format.raw/*291.18*/(""");

                lastResultsUpdateTime = resultsTime;
            """),format.raw/*294.13*/("""}"""),format.raw/*294.14*/("""

            """),format.raw/*296.13*/("""//Finally: Currently selected result
            if(selectedCandidateIdx != null)"""),format.raw/*297.45*/("""{"""),format.raw/*297.46*/("""
                """),format.raw/*298.17*/("""//Get JSON for components
                $.get("/arbiter/candidateInfo/"+selectedCandidateIdx,function(data)"""),format.raw/*299.84*/("""{"""),format.raw/*299.85*/("""
                    """),format.raw/*300.21*/("""var str = JSON.stringify(data);

                    var resultsViewDiv = $('#resultsviewdiv');
                    resultsViewDiv.html('');

                    var component = Component.getComponent(str);
                    component.render(resultsViewDiv);
                """),format.raw/*307.17*/("""}"""),format.raw/*307.18*/(""");
            """),format.raw/*308.13*/("""}"""),format.raw/*308.14*/("""
        """),format.raw/*309.9*/("""}"""),format.raw/*309.10*/(""")
    """),format.raw/*310.5*/("""}"""),format.raw/*310.6*/("""

    """),format.raw/*312.5*/("""function createTable(tableObj,tableId,appendTo)"""),format.raw/*312.52*/("""{"""),format.raw/*312.53*/("""
        """),format.raw/*313.9*/("""//Expect RenderableComponentTable
        var header = tableObj['header'];
        var values = tableObj['table'];
        var title = tableObj['title'];
        var nRows = (values ? values.length : 0);

        if(title)"""),format.raw/*319.18*/("""{"""),format.raw/*319.19*/("""
            """),format.raw/*320.13*/("""appendTo.append("<h5>"+title+"</h5>");
        """),format.raw/*321.9*/("""}"""),format.raw/*321.10*/("""

        """),format.raw/*323.9*/("""var table;
        if(tableId) table = $("<table id=\"" + tableId + "\" class=\"renderableComponentTable\">");
        else table = $("<table class=\"renderableComponentTable\">");
        if(header)"""),format.raw/*326.19*/("""{"""),format.raw/*326.20*/("""
            """),format.raw/*327.13*/("""var headerRow = $("<tr>");
            var len = header.length;
            for( var i=0; i<len; i++ )"""),format.raw/*329.39*/("""{"""),format.raw/*329.40*/("""
                """),format.raw/*330.17*/("""headerRow.append($("<th>" + header[i] + "</th>"));
            """),format.raw/*331.13*/("""}"""),format.raw/*331.14*/("""
            """),format.raw/*332.13*/("""headerRow.append($("</tr>"));
            table.append(headerRow);
        """),format.raw/*334.9*/("""}"""),format.raw/*334.10*/("""

        """),format.raw/*336.9*/("""if(values)"""),format.raw/*336.19*/("""{"""),format.raw/*336.20*/("""
            """),format.raw/*337.13*/("""for( var i=0; i<nRows; i++ )"""),format.raw/*337.41*/("""{"""),format.raw/*337.42*/("""
                """),format.raw/*338.17*/("""var row = $("<tr>");
                var rowValues = values[i];
                var len = rowValues.length;
                for( var j=0; j<len; j++ )"""),format.raw/*341.43*/("""{"""),format.raw/*341.44*/("""
                    """),format.raw/*342.21*/("""row.append($('<td>'+rowValues[j]+'</td>'));
                """),format.raw/*343.17*/("""}"""),format.raw/*343.18*/("""
                """),format.raw/*344.17*/("""row.append($("</tr>"));
                table.append(row);
            """),format.raw/*346.13*/("""}"""),format.raw/*346.14*/("""
        """),format.raw/*347.9*/("""}"""),format.raw/*347.10*/("""

        """),format.raw/*349.9*/("""table.append($("</table>"));
        appendTo.append(table);
    """),format.raw/*351.5*/("""}"""),format.raw/*351.6*/("""

    """),format.raw/*353.5*/("""function drawResultTable()"""),format.raw/*353.31*/("""{"""),format.raw/*353.32*/("""
        """),format.raw/*354.9*/("""//Remove all elements from the table body
        var tableBody = $('#resultsTableBody');
        tableBody.empty();

        //Recreate the table header, with appropriate sort order:
        var tableHeader = $('#resultsTableHeader');
        tableHeader.empty();
        var headerRow = $("<tr />");
        var char = (resultTableSortOrder== "ascending" ? "&blacktriangledown;" : "&blacktriangle;");
        if(resultTableSortIndex == 0) headerRow.append("$(<th>ID &nbsp; " + char + "</th>");
        else headerRow.append("$(<th>ID</th>");
        if(resultTableSortIndex == 1) headerRow.append("$(<th>Score &nbsp; " + char + "</th>");
        else headerRow.append("$(<th>Score</th>");
        if(resultTableSortIndex == 2) headerRow.append("$(<th>Status &nbsp; " + char + "</th>");
        else headerRow.append("$(<th>Status</th>");
        tableHeader.append(headerRow);


        //Sort rows, and insert into table:
        var sorted;
        if(resultTableSortIndex == 0) sorted = resultsTableContent.sort(compareResultsIndex);
        else if(resultTableSortIndex == 1) sorted = resultsTableContent.sort(compareScores);
        else sorted = resultsTableContent.sort(compareStatus);

        var len = (!resultsTableContent ? 0 : resultsTableContent.length);
        for(var i=0; i<len; i++)"""),format.raw/*379.33*/("""{"""),format.raw/*379.34*/("""
            """),format.raw/*380.13*/("""var row;
            if(selectedCandidateIdx == sorted[i][0])"""),format.raw/*381.53*/("""{"""),format.raw/*381.54*/("""
                """),format.raw/*382.17*/("""//Selected row
                row = $('<tr class="resultTableRowSelected" id="rTbl-' + sorted[i][0] + '"/>');
            """),format.raw/*384.13*/("""}"""),format.raw/*384.14*/(""" """),format.raw/*384.15*/("""else """),format.raw/*384.20*/("""{"""),format.raw/*384.21*/("""
                """),format.raw/*385.17*/("""//Normal row
                row = $('<tr class="resultTableRow" id="rTbl-' + sorted[i][0] + '"/>');
            """),format.raw/*387.13*/("""}"""),format.raw/*387.14*/("""
            """),format.raw/*388.13*/("""row.append($("<td>" + sorted[i][0] + "</td>"));
            var score = sorted[i][1];
            row.append($("<td>" + ((!score || score == "null") ? "-" : score) + "</td>"));
            row.append($("<td>" + sorted[i][2] + "</td>"));
            tableBody.append(row);
        """),format.raw/*393.9*/("""}"""),format.raw/*393.10*/("""
    """),format.raw/*394.5*/("""}"""),format.raw/*394.6*/("""

    """),format.raw/*396.5*/("""//Compare function for results, based on sort order
    function compareResultsIndex(a, b)"""),format.raw/*397.39*/("""{"""),format.raw/*397.40*/("""
        """),format.raw/*398.9*/("""return (resultTableSortOrder == "ascending" ? a[0] - b[0] : b[0] - a[0]);
    """),format.raw/*399.5*/("""}"""),format.raw/*399.6*/("""
    """),format.raw/*400.5*/("""function compareScores(a,b)"""),format.raw/*400.32*/("""{"""),format.raw/*400.33*/("""
        """),format.raw/*401.9*/("""//TODO Not always numbers...
        if(resultTableSortOrder == "ascending")"""),format.raw/*402.48*/("""{"""),format.raw/*402.49*/("""
            """),format.raw/*403.13*/("""if(a[1] == "NaN")"""),format.raw/*403.30*/("""{"""),format.raw/*403.31*/("""
                """),format.raw/*404.17*/("""return 1;
            """),format.raw/*405.13*/("""}"""),format.raw/*405.14*/(""" """),format.raw/*405.15*/("""else if(b[1] == "NaN")"""),format.raw/*405.37*/("""{"""),format.raw/*405.38*/("""
                """),format.raw/*406.17*/("""return -1;
            """),format.raw/*407.13*/("""}"""),format.raw/*407.14*/("""
            """),format.raw/*408.13*/("""return a[1] - b[1];
        """),format.raw/*409.9*/("""}"""),format.raw/*409.10*/(""" """),format.raw/*409.11*/("""else """),format.raw/*409.16*/("""{"""),format.raw/*409.17*/("""
            """),format.raw/*410.13*/("""if(a[1] == "NaN")"""),format.raw/*410.30*/("""{"""),format.raw/*410.31*/("""
                """),format.raw/*411.17*/("""return -1;
            """),format.raw/*412.13*/("""}"""),format.raw/*412.14*/(""" """),format.raw/*412.15*/("""else if(b[1] == "NaN")"""),format.raw/*412.37*/("""{"""),format.raw/*412.38*/("""
                """),format.raw/*413.17*/("""return 1;
            """),format.raw/*414.13*/("""}"""),format.raw/*414.14*/("""
            """),format.raw/*415.13*/("""return b[1] - a[1];
        """),format.raw/*416.9*/("""}"""),format.raw/*416.10*/("""
    """),format.raw/*417.5*/("""}"""),format.raw/*417.6*/("""
    """),format.raw/*418.5*/("""function compareStatus(a,b)"""),format.raw/*418.32*/("""{"""),format.raw/*418.33*/("""
        """),format.raw/*419.9*/("""//TODO: secondary sort on... score? index?
        if(resultTableSortOrder == "ascending")"""),format.raw/*420.48*/("""{"""),format.raw/*420.49*/("""
            """),format.raw/*421.13*/("""return (a[2] < b[2] ? -1 : (a[2] > b[2] ? 1 : 0));
        """),format.raw/*422.9*/("""}"""),format.raw/*422.10*/(""" """),format.raw/*422.11*/("""else """),format.raw/*422.16*/("""{"""),format.raw/*422.17*/("""
            """),format.raw/*423.13*/("""return (a[2] < b[2] ? 1 : (a[2] > b[2] ? -1 : 0));
        """),format.raw/*424.9*/("""}"""),format.raw/*424.10*/("""
    """),format.raw/*425.5*/("""}"""),format.raw/*425.6*/("""

    """),format.raw/*427.5*/("""//Do a HTTP request on the specified path, parse and insert into the provided element
    function loadCandidateDetails(path, elementToAppendTo)"""),format.raw/*428.59*/("""{"""),format.raw/*428.60*/("""
        """),format.raw/*429.9*/("""$.get(path, function (data) """),format.raw/*429.37*/("""{"""),format.raw/*429.38*/("""
            """),format.raw/*430.13*/("""var str = JSON.stringify(data);
            var component = Component.getComponent(str);
            component.render(elementToAppendTo);
        """),format.raw/*433.9*/("""}"""),format.raw/*433.10*/(""");
    """),format.raw/*434.5*/("""}"""),format.raw/*434.6*/("""



    """),format.raw/*438.5*/("""//Sorting by column: Intercept click events on table header
    $(function()"""),format.raw/*439.17*/("""{"""),format.raw/*439.18*/("""
        """),format.raw/*440.9*/("""$("#resultsTableHeader").delegate("th", "click", function(e) """),format.raw/*440.70*/("""{"""),format.raw/*440.71*/("""
            """),format.raw/*441.13*/("""//console.log("Header clicked on at: " + $(e.currentTarget).index() + " - " + $(e.currentTarget).html());
            //Update the sort order for the table:
            var clickIndex = $(e.currentTarget).index();
            if(clickIndex == resultTableSortIndex)"""),format.raw/*444.51*/("""{"""),format.raw/*444.52*/("""
                """),format.raw/*445.17*/("""//Switch sort order: ascending -> descending or descending -> ascending
                if(resultTableSortOrder == "ascending")"""),format.raw/*446.56*/("""{"""),format.raw/*446.57*/("""
                    """),format.raw/*447.21*/("""resultTableSortOrder = "descending";
                """),format.raw/*448.17*/("""}"""),format.raw/*448.18*/(""" """),format.raw/*448.19*/("""else """),format.raw/*448.24*/("""{"""),format.raw/*448.25*/("""
                    """),format.raw/*449.21*/("""resultTableSortOrder = "ascending";
                """),format.raw/*450.17*/("""}"""),format.raw/*450.18*/("""
            """),format.raw/*451.13*/("""}"""),format.raw/*451.14*/(""" """),format.raw/*451.15*/("""else """),format.raw/*451.20*/("""{"""),format.raw/*451.21*/("""
                """),format.raw/*452.17*/("""//Sort on column, ascending:
                resultTableSortIndex = clickIndex;
                resultTableSortOrder = "ascending";
            """),format.raw/*455.13*/("""}"""),format.raw/*455.14*/("""

            """),format.raw/*457.13*/("""//Clear record of expanded rows
            expandedRowsCandidateIDs = [];

            //Redraw table
            drawResultTable();
        """),format.raw/*462.9*/("""}"""),format.raw/*462.10*/(""");
    """),format.raw/*463.5*/("""}"""),format.raw/*463.6*/(""");

    //Displaying model/candidate details: Intercept click events on table rows -> toggle selected, fire off update
    $(function()"""),format.raw/*466.17*/("""{"""),format.raw/*466.18*/("""
        """),format.raw/*467.9*/("""$("#resultsTableBody").delegate("tr", "click", function(e)"""),format.raw/*467.67*/("""{"""),format.raw/*467.68*/("""
            """),format.raw/*468.13*/("""var id = this.id;   //Expect: rTbl-X  where X is some index
            var dashIdx = id.indexOf("-");
            var candidateID = Number(id.substring(dashIdx+1));
//            console.log("Clicked row: " + this.id + " with class: " + this.className + ", candidateId = " + candidateID);

            if(this.className == "resultTableRow")"""),format.raw/*473.51*/("""{"""),format.raw/*473.52*/("""
                """),format.raw/*474.17*/("""//Set selected model
                selectedCandidateIdx = candidateID;

                //Fire off update
                doUpdate();
            """),format.raw/*479.13*/("""}"""),format.raw/*479.14*/("""
        """),format.raw/*480.9*/("""}"""),format.raw/*480.10*/(""");
    """),format.raw/*481.5*/("""}"""),format.raw/*481.6*/(""");

    function selectNewSession()"""),format.raw/*483.32*/("""{"""),format.raw/*483.33*/("""
        """),format.raw/*484.9*/("""var selector = $("#sessionSelect");
        var currSelected = selector.val();

        if(currSelected)"""),format.raw/*487.25*/("""{"""),format.raw/*487.26*/("""
            """),format.raw/*488.13*/("""$.ajax("""),format.raw/*488.20*/("""{"""),format.raw/*488.21*/("""
                """),format.raw/*489.17*/("""url: "/arbiter/sessions/set/" + currSelected,
                async: true,
                error: function (query, status, error) """),format.raw/*491.56*/("""{"""),format.raw/*491.57*/("""
                    """),format.raw/*492.21*/("""console.log("Error setting session: " + error);
                """),format.raw/*493.17*/("""}"""),format.raw/*493.18*/(""",
                success: function (data) """),format.raw/*494.42*/("""{"""),format.raw/*494.43*/("""
                    """),format.raw/*495.21*/("""//Update UI immediately
                    doUpdate();
                """),format.raw/*497.17*/("""}"""),format.raw/*497.18*/("""
            """),format.raw/*498.13*/("""}"""),format.raw/*498.14*/(""");
        """),format.raw/*499.9*/("""}"""),format.raw/*499.10*/("""
    """),format.raw/*500.5*/("""}"""),format.raw/*500.6*/("""

"""),format.raw/*502.1*/("""</script>
<script>
    $(function () """),format.raw/*504.19*/("""{"""),format.raw/*504.20*/("""
        """),format.raw/*505.9*/("""$("#accordion").accordion("""),format.raw/*505.35*/("""{"""),format.raw/*505.36*/("""
            """),format.raw/*506.13*/("""collapsible: true,
            heightStyle: "content"
        """),format.raw/*508.9*/("""}"""),format.raw/*508.10*/(""");
    """),format.raw/*509.5*/("""}"""),format.raw/*509.6*/(""");
    $(function () """),format.raw/*510.19*/("""{"""),format.raw/*510.20*/("""
        """),format.raw/*511.9*/("""$("#accordion2").accordion("""),format.raw/*511.36*/("""{"""),format.raw/*511.37*/("""
            """),format.raw/*512.13*/("""collapsible: true,
            heightStyle: "content"
        """),format.raw/*514.9*/("""}"""),format.raw/*514.10*/(""");
    """),format.raw/*515.5*/("""}"""),format.raw/*515.6*/(""");
    $(function () """),format.raw/*516.19*/("""{"""),format.raw/*516.20*/("""
        """),format.raw/*517.9*/("""$("#accordion3").accordion("""),format.raw/*517.36*/("""{"""),format.raw/*517.37*/("""
            """),format.raw/*518.13*/("""collapsible: true,
            heightStyle: "content"
        """),format.raw/*520.9*/("""}"""),format.raw/*520.10*/(""");
    """),format.raw/*521.5*/("""}"""),format.raw/*521.6*/(""");
</script>
        <table style="width: 100%;
            padding: 5px;" class="hd">
            <tbody>
                <tr style="height: 40px">
                    <td> <div style="width: 40px;
                        height: 40px;
                        float: left"></div>
                        <div style="height: 40px;
                            float: left;
                            padding-top: 10px">Deeplearning4J - Arbiter UI</div></td>
                    <td>
                        <div id="sessionSelectDiv" style="display:none; float:left; font-size: 10pt; color:black">
                            <span style="color: white">Session:</span>
                        <select id="sessionSelect" onchange='selectNewSession()'>
                            <option>(Session ID)</option>
                        </select>
                    </div>
                    </td>
                </tr>
            </tbody>
        </table>


        <div style="width: 1200px;
            margin-left: auto;
            margin-right: auto;">
            <div class="outerelements" id="status">
                <div id="accordion" class="hcol2">
                    <h3 class="hcol2 headingcolor ui-accordion-header">Summary</h3>
                    <div class="statusdiv" id="statusdiv">
            </div>
                </div>
            </div>

            <div class="outerelements" id="settings">
                <div id="accordion2">
                    <h3 class="ui-accordion-header headingcolor">Optimization Settings</h3>
                    <div class="settingsdiv" id="settingsdiv"></div>
                </div>
            </div>


            <div class="outerelements" id="results">
                <div class="resultsHeadingDiv">Results</div>
                <div class="resultsdiv" id="resultsdiv">
                    <table style="width: 100%" id="resultsTable" class="resultsTable">
                        <col width="33%">
                        <col width="33%">
                        <col width="34%">
                        <thead id="resultsTableHeader"></thead>
                        <tbody id="resultsTableBody"></tbody>
                    </table>
                </div>
            </div>

            <div class="outerelements" id="resultview">
                <div id="accordion3">
                    <h3 class="ui-accordion-header headingcolor">Selected Result</h3>
                    <div class="resultsviewdiv" id="resultsviewdiv"></div>
                </div>
            </div>
        </div>
    </body>
</html>"""))
      }
    }
  }

  def render(): play.twirl.api.HtmlFormat.Appendable = apply()

  def f:(() => play.twirl.api.HtmlFormat.Appendable) = () => apply()

  def ref: this.type = this

}


}

/**/
object ArbiterUI extends ArbiterUI_Scope0.ArbiterUI
              /*
                  -- GENERATED --
                  DATE: Thu Sep 05 18:53:38 AEST 2019
                  SOURCE: C:/DL4J/Git/deeplearning4j/arbiter/arbiter-ui/src/main/views/org/deeplearning4j/arbiter/ui/views/ArbiterUI.scala.html
                  HASH: aa565fc00b748d09bde27d54caef529f2933efbb
                  MATRIX: 647->0|1821->1146|1850->1147|1892->1161|2123->1365|2152->1366|2191->1378|2230->1389|2259->1390|2301->1404|2402->1478|2431->1479|2470->1491|2507->1500|2536->1501|2578->1515|2641->1551|2670->1552|2709->1564|2740->1567|2769->1568|2811->1582|3046->1790|3075->1791|3114->1803|3145->1806|3174->1807|3216->1821|3453->2031|3482->2032|3521->2044|3555->2050|3584->2051|3626->2065|3730->2142|3759->2143|3798->2155|3845->2174|3874->2175|3916->2189|4088->2334|4117->2335|4156->2347|4252->2415|4281->2416|4323->2430|4465->2545|4494->2546|4533->2558|4583->2580|4612->2581|4654->2595|4825->2739|4854->2740|4893->2752|4943->2774|4972->2775|5014->2789|5144->2892|5173->2893|5212->2905|5372->3037|5401->3038|5443->3052|5596->3178|5625->3179|5662->3189|5719->3218|5748->3219|5791->3233|6048->3462|6078->3463|6118->3475|6205->3533|6235->3534|6278->3548|6372->3614|6402->3615|6442->3627|6580->3736|6610->3737|6653->3751|6733->3803|6763->3804|6803->3816|6851->3835|6881->3836|6924->3850|7507->4405|7537->4406|7577->4418|7624->4436|7654->4437|7697->4451|7756->4482|7786->4483|7826->4495|7879->4519|7909->4520|7952->4534|8011->4565|8041->4566|8081->4578|8211->4678|8242->4679|8285->4693|8723->5103|8753->5104|8793->5116|8855->5149|8885->5150|8928->5164|9259->5467|9289->5468|9329->5480|9391->5513|9421->5514|9464->5528|9575->5611|9605->5612|9643->5622|9695->5645|9725->5646|9768->5660|9886->5750|9916->5751|9954->5761|9994->5772|10024->5773|10067->5787|10160->5852|10190->5853|10230->5865|11496->7102|11526->7103|11564->7113|11695->7215|11725->7216|11768->7230|12332->7765|12362->7766|12409->7784|12552->7898|12582->7899|12674->7962|12704->7963|12751->7981|12863->8064|12893->8065|12944->8087|13201->8315|13231->8316|13286->8342|13341->8368|13371->8369|13430->8399|13505->8445|13535->8446|13590->8472|13713->8566|13743->8567|13796->8591|13989->8755|14019->8756|14049->8758|14177->8857|14207->8858|14429->9051|14459->9052|14506->9070|14640->9175|14670->9176|14721->9198|15033->9481|15063->9482|15162->9552|15192->9553|15237->9569|15361->9664|15391->9665|15438->9683|15548->9764|15578->9765|15629->9787|15924->10053|15954->10054|16057->10128|16087->10129|16132->10145|16287->10271|16317->10272|16366->10292|16501->10398|16531->10399|16582->10421|16770->10580|16800->10581|16901->10653|16931->10654|16976->10670|17087->10752|17117->10753|17164->10771|17303->10881|17333->10882|17384->10904|17697->11188|17727->11189|17772->11205|17802->11206|17840->11216|17870->11217|17905->11224|17934->11225|17970->11233|18046->11280|18076->11281|18114->11291|18371->11519|18401->11520|18444->11534|18520->11582|18550->11583|18590->11595|18821->11797|18851->11798|18894->11812|19027->11916|19057->11917|19104->11935|19197->11999|19227->12000|19270->12014|19375->12091|19405->12092|19445->12104|19484->12114|19514->12115|19557->12129|19614->12157|19644->12158|19691->12176|19873->12329|19903->12330|19954->12352|20044->12413|20074->12414|20121->12432|20223->12505|20253->12506|20291->12516|20321->12517|20361->12529|20456->12596|20485->12597|20521->12605|20576->12631|20606->12632|20644->12642|22001->13970|22031->13971|22074->13985|22165->14047|22195->14048|22242->14066|22396->14191|22426->14192|22456->14193|22490->14198|22520->14199|22567->14217|22711->14332|22741->14333|22784->14347|23097->14632|23127->14633|23161->14639|23190->14640|23226->14648|23346->14739|23376->14740|23414->14750|23521->14829|23550->14830|23584->14836|23640->14863|23670->14864|23708->14874|23814->14951|23844->14952|23887->14966|23933->14983|23963->14984|24010->15002|24062->15025|24092->15026|24122->15027|24173->15049|24203->15050|24250->15068|24303->15092|24333->15093|24376->15107|24433->15136|24463->15137|24493->15138|24527->15143|24557->15144|24600->15158|24646->15175|24676->15176|24723->15194|24776->15218|24806->15219|24836->15220|24887->15242|24917->15243|24964->15261|25016->15284|25046->15285|25089->15299|25146->15328|25176->15329|25210->15335|25239->15336|25273->15342|25329->15369|25359->15370|25397->15380|25517->15471|25547->15472|25590->15486|25678->15546|25708->15547|25738->15548|25772->15553|25802->15554|25845->15568|25933->15628|25963->15629|25997->15635|26026->15636|26062->15644|26236->15789|26266->15790|26304->15800|26361->15828|26391->15829|26434->15843|26611->15992|26641->15993|26677->16001|26706->16002|26746->16014|26852->16091|26882->16092|26920->16102|27010->16163|27040->16164|27083->16178|27379->16445|27409->16446|27456->16464|27613->16592|27643->16593|27694->16615|27777->16669|27807->16670|27837->16671|27871->16676|27901->16677|27952->16699|28034->16752|28064->16753|28107->16767|28137->16768|28167->16769|28201->16774|28231->16775|28278->16793|28454->16940|28484->16941|28529->16957|28704->17104|28734->17105|28770->17113|28799->17114|28966->17252|28996->17253|29034->17263|29121->17321|29151->17322|29194->17336|29569->17682|29599->17683|29646->17701|29828->17854|29858->17855|29896->17865|29926->17866|29962->17874|29991->17875|30057->17912|30087->17913|30125->17923|30261->18030|30291->18031|30334->18045|30370->18052|30400->18053|30447->18071|30608->18203|30638->18204|30689->18226|30783->18291|30813->18292|30886->18336|30916->18337|30967->18359|31070->18433|31100->18434|31143->18448|31173->18449|31213->18461|31243->18462|31277->18468|31306->18469|31338->18473|31406->18512|31436->18513|31474->18523|31529->18549|31559->18550|31602->18564|31694->18628|31724->18629|31760->18637|31789->18638|31840->18660|31870->18661|31908->18671|31964->18698|31994->18699|32037->18713|32129->18777|32159->18778|32195->18786|32224->18787|32275->18809|32305->18810|32343->18820|32399->18847|32429->18848|32472->18862|32564->18926|32594->18927|32630->18935|32659->18936
                  LINES: 25->1|51->27|51->27|52->28|58->34|58->34|60->36|60->36|60->36|61->37|64->40|64->40|66->42|66->42|66->42|67->43|68->44|68->44|70->46|70->46|70->46|71->47|77->53|77->53|79->55|79->55|79->55|80->56|86->62|86->62|88->64|88->64|88->64|89->65|91->67|91->67|93->69|93->69|93->69|94->70|98->74|98->74|100->76|100->76|100->76|101->77|103->79|103->79|105->81|105->81|105->81|106->82|110->86|110->86|112->88|112->88|112->88|113->89|116->92|116->92|118->94|119->95|119->95|120->96|122->98|122->98|123->99|123->99|123->99|124->100|129->105|129->105|131->107|132->108|132->108|133->109|135->111|135->111|137->113|138->114|138->114|139->115|140->116|140->116|142->118|142->118|142->118|143->119|159->135|159->135|161->137|161->137|161->137|162->138|163->139|163->139|165->141|165->141|165->141|166->142|167->143|167->143|169->145|169->145|169->145|170->146|178->154|178->154|180->156|180->156|180->156|181->157|187->163|187->163|189->165|190->166|190->166|191->167|194->170|194->170|195->171|195->171|195->171|196->172|199->175|199->175|200->176|200->176|200->176|201->177|203->179|203->179|205->181|234->210|234->210|235->211|236->212|236->212|237->213|246->222|246->222|247->223|249->225|249->225|251->227|251->227|252->228|254->230|254->230|255->231|261->237|261->237|262->238|262->238|262->238|263->239|264->240|264->240|265->241|266->242|266->242|268->244|270->246|270->246|271->247|272->248|272->248|277->253|277->253|278->254|279->255|279->255|280->256|286->262|286->262|289->265|289->265|291->267|292->268|292->268|293->269|294->270|294->270|295->271|302->278|302->278|305->281|305->281|307->283|308->284|308->284|310->286|311->287|311->287|312->288|315->291|315->291|318->294|318->294|320->296|321->297|321->297|322->298|323->299|323->299|324->300|331->307|331->307|332->308|332->308|333->309|333->309|334->310|334->310|336->312|336->312|336->312|337->313|343->319|343->319|344->320|345->321|345->321|347->323|350->326|350->326|351->327|353->329|353->329|354->330|355->331|355->331|356->332|358->334|358->334|360->336|360->336|360->336|361->337|361->337|361->337|362->338|365->341|365->341|366->342|367->343|367->343|368->344|370->346|370->346|371->347|371->347|373->349|375->351|375->351|377->353|377->353|377->353|378->354|403->379|403->379|404->380|405->381|405->381|406->382|408->384|408->384|408->384|408->384|408->384|409->385|411->387|411->387|412->388|417->393|417->393|418->394|418->394|420->396|421->397|421->397|422->398|423->399|423->399|424->400|424->400|424->400|425->401|426->402|426->402|427->403|427->403|427->403|428->404|429->405|429->405|429->405|429->405|429->405|430->406|431->407|431->407|432->408|433->409|433->409|433->409|433->409|433->409|434->410|434->410|434->410|435->411|436->412|436->412|436->412|436->412|436->412|437->413|438->414|438->414|439->415|440->416|440->416|441->417|441->417|442->418|442->418|442->418|443->419|444->420|444->420|445->421|446->422|446->422|446->422|446->422|446->422|447->423|448->424|448->424|449->425|449->425|451->427|452->428|452->428|453->429|453->429|453->429|454->430|457->433|457->433|458->434|458->434|462->438|463->439|463->439|464->440|464->440|464->440|465->441|468->444|468->444|469->445|470->446|470->446|471->447|472->448|472->448|472->448|472->448|472->448|473->449|474->450|474->450|475->451|475->451|475->451|475->451|475->451|476->452|479->455|479->455|481->457|486->462|486->462|487->463|487->463|490->466|490->466|491->467|491->467|491->467|492->468|497->473|497->473|498->474|503->479|503->479|504->480|504->480|505->481|505->481|507->483|507->483|508->484|511->487|511->487|512->488|512->488|512->488|513->489|515->491|515->491|516->492|517->493|517->493|518->494|518->494|519->495|521->497|521->497|522->498|522->498|523->499|523->499|524->500|524->500|526->502|528->504|528->504|529->505|529->505|529->505|530->506|532->508|532->508|533->509|533->509|534->510|534->510|535->511|535->511|535->511|536->512|538->514|538->514|539->515|539->515|540->516|540->516|541->517|541->517|541->517|542->518|544->520|544->520|545->521|545->521
                  -- GENERATED --
              */
          