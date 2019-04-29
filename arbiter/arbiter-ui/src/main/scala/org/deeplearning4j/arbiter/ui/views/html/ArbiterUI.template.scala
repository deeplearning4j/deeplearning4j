
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
        <link id="bootstrap-style" href="/assets/webjars/bootstrap/2.3.2/css/bootstrap.min.css" rel="stylesheet">
        <script src="/assets/webjars/jquery/2.2.0/jquery.min.js"></script>
        <link href="/assets/webjars/jquery-ui/1.10.2/themes/base/jquery-ui.css" rel="stylesheet">
        <script src="/assets/webjars/jquery-ui/1.10.2/ui/minified/jquery-ui.min.js"></script>
        <script src="/assets/webjars/d3js/3.3.5/d3.min.js" charset="utf-8"></script>
        <script src="/assets/webjars/bootstrap/2.3.2/js/bootstrap.min.js"></script>
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
                  DATE: Wed Apr 24 12:41:30 AEST 2019
                  SOURCE: C:/DL4J/Git/deeplearning4j/arbiter/arbiter-ui/src/main/views/org/deeplearning4j/arbiter/ui/views/ArbiterUI.scala.html
                  HASH: e64cc745763d17ded8a68f9ec29d5afba67aac6b
                  MATRIX: 647->0|1821->1146|1850->1147|1892->1161|2123->1365|2152->1366|2191->1378|2230->1389|2259->1390|2301->1404|2402->1478|2431->1479|2470->1491|2507->1500|2536->1501|2578->1515|2641->1551|2670->1552|2709->1564|2740->1567|2769->1568|2811->1582|3046->1790|3075->1791|3114->1803|3145->1806|3174->1807|3216->1821|3453->2031|3482->2032|3521->2044|3555->2050|3584->2051|3626->2065|3730->2142|3759->2143|3798->2155|3845->2174|3874->2175|3916->2189|4088->2334|4117->2335|4156->2347|4252->2415|4281->2416|4323->2430|4465->2545|4494->2546|4533->2558|4583->2580|4612->2581|4654->2595|4825->2739|4854->2740|4893->2752|4943->2774|4972->2775|5014->2789|5144->2892|5173->2893|5212->2905|5372->3037|5401->3038|5443->3052|5596->3178|5625->3179|5662->3189|5719->3218|5748->3219|5791->3233|6048->3462|6078->3463|6118->3475|6205->3533|6235->3534|6278->3548|6372->3614|6402->3615|6442->3627|6580->3736|6610->3737|6653->3751|6733->3803|6763->3804|6803->3816|6851->3835|6881->3836|6924->3850|7507->4405|7537->4406|7577->4418|7624->4436|7654->4437|7697->4451|7756->4482|7786->4483|7826->4495|7879->4519|7909->4520|7952->4534|8011->4565|8041->4566|8081->4578|8211->4678|8242->4679|8285->4693|8723->5103|8753->5104|8793->5116|8855->5149|8885->5150|8928->5164|9259->5467|9289->5468|9329->5480|9391->5513|9421->5514|9464->5528|9575->5611|9605->5612|9643->5622|9695->5645|9725->5646|9768->5660|9886->5750|9916->5751|9954->5761|9994->5772|10024->5773|10067->5787|10160->5852|10190->5853|10230->5865|11486->7092|11516->7093|11554->7103|11685->7205|11715->7206|11758->7220|12322->7755|12352->7756|12399->7774|12542->7888|12572->7889|12664->7952|12694->7953|12741->7971|12853->8054|12883->8055|12934->8077|13191->8305|13221->8306|13276->8332|13331->8358|13361->8359|13420->8389|13495->8435|13525->8436|13580->8462|13703->8556|13733->8557|13786->8581|13979->8745|14009->8746|14039->8748|14167->8847|14197->8848|14419->9041|14449->9042|14496->9060|14630->9165|14660->9166|14711->9188|15023->9471|15053->9472|15152->9542|15182->9543|15227->9559|15351->9654|15381->9655|15428->9673|15538->9754|15568->9755|15619->9777|15914->10043|15944->10044|16047->10118|16077->10119|16122->10135|16277->10261|16307->10262|16356->10282|16491->10388|16521->10389|16572->10411|16760->10570|16790->10571|16891->10643|16921->10644|16966->10660|17077->10742|17107->10743|17154->10761|17293->10871|17323->10872|17374->10894|17687->11178|17717->11179|17762->11195|17792->11196|17830->11206|17860->11207|17895->11214|17924->11215|17960->11223|18036->11270|18066->11271|18104->11281|18361->11509|18391->11510|18434->11524|18510->11572|18540->11573|18580->11585|18811->11787|18841->11788|18884->11802|19017->11906|19047->11907|19094->11925|19187->11989|19217->11990|19260->12004|19365->12081|19395->12082|19435->12094|19474->12104|19504->12105|19547->12119|19604->12147|19634->12148|19681->12166|19863->12319|19893->12320|19944->12342|20034->12403|20064->12404|20111->12422|20213->12495|20243->12496|20281->12506|20311->12507|20351->12519|20446->12586|20475->12587|20511->12595|20566->12621|20596->12622|20634->12632|21991->13960|22021->13961|22064->13975|22155->14037|22185->14038|22232->14056|22386->14181|22416->14182|22446->14183|22480->14188|22510->14189|22557->14207|22701->14322|22731->14323|22774->14337|23087->14622|23117->14623|23151->14629|23180->14630|23216->14638|23336->14729|23366->14730|23404->14740|23511->14819|23540->14820|23574->14826|23630->14853|23660->14854|23698->14864|23804->14941|23834->14942|23877->14956|23923->14973|23953->14974|24000->14992|24052->15015|24082->15016|24112->15017|24163->15039|24193->15040|24240->15058|24293->15082|24323->15083|24366->15097|24423->15126|24453->15127|24483->15128|24517->15133|24547->15134|24590->15148|24636->15165|24666->15166|24713->15184|24766->15208|24796->15209|24826->15210|24877->15232|24907->15233|24954->15251|25006->15274|25036->15275|25079->15289|25136->15318|25166->15319|25200->15325|25229->15326|25263->15332|25319->15359|25349->15360|25387->15370|25507->15461|25537->15462|25580->15476|25668->15536|25698->15537|25728->15538|25762->15543|25792->15544|25835->15558|25923->15618|25953->15619|25987->15625|26016->15626|26052->15634|26226->15779|26256->15780|26294->15790|26351->15818|26381->15819|26424->15833|26601->15982|26631->15983|26667->15991|26696->15992|26736->16004|26842->16081|26872->16082|26910->16092|27000->16153|27030->16154|27073->16168|27369->16435|27399->16436|27446->16454|27603->16582|27633->16583|27684->16605|27767->16659|27797->16660|27827->16661|27861->16666|27891->16667|27942->16689|28024->16742|28054->16743|28097->16757|28127->16758|28157->16759|28191->16764|28221->16765|28268->16783|28444->16930|28474->16931|28519->16947|28694->17094|28724->17095|28760->17103|28789->17104|28956->17242|28986->17243|29024->17253|29111->17311|29141->17312|29184->17326|29559->17672|29589->17673|29636->17691|29818->17844|29848->17845|29886->17855|29916->17856|29952->17864|29981->17865|30047->17902|30077->17903|30115->17913|30251->18020|30281->18021|30324->18035|30360->18042|30390->18043|30437->18061|30598->18193|30628->18194|30679->18216|30773->18281|30803->18282|30876->18326|30906->18327|30957->18349|31060->18423|31090->18424|31133->18438|31163->18439|31203->18451|31233->18452|31267->18458|31296->18459|31328->18463|31396->18502|31426->18503|31464->18513|31519->18539|31549->18540|31592->18554|31684->18618|31714->18619|31750->18627|31779->18628|31830->18650|31860->18651|31898->18661|31954->18688|31984->18689|32027->18703|32119->18767|32149->18768|32185->18776|32214->18777|32265->18799|32295->18800|32333->18810|32389->18837|32419->18838|32462->18852|32554->18916|32584->18917|32620->18925|32649->18926
                  LINES: 25->1|51->27|51->27|52->28|58->34|58->34|60->36|60->36|60->36|61->37|64->40|64->40|66->42|66->42|66->42|67->43|68->44|68->44|70->46|70->46|70->46|71->47|77->53|77->53|79->55|79->55|79->55|80->56|86->62|86->62|88->64|88->64|88->64|89->65|91->67|91->67|93->69|93->69|93->69|94->70|98->74|98->74|100->76|100->76|100->76|101->77|103->79|103->79|105->81|105->81|105->81|106->82|110->86|110->86|112->88|112->88|112->88|113->89|116->92|116->92|118->94|119->95|119->95|120->96|122->98|122->98|123->99|123->99|123->99|124->100|129->105|129->105|131->107|132->108|132->108|133->109|135->111|135->111|137->113|138->114|138->114|139->115|140->116|140->116|142->118|142->118|142->118|143->119|159->135|159->135|161->137|161->137|161->137|162->138|163->139|163->139|165->141|165->141|165->141|166->142|167->143|167->143|169->145|169->145|169->145|170->146|178->154|178->154|180->156|180->156|180->156|181->157|187->163|187->163|189->165|190->166|190->166|191->167|194->170|194->170|195->171|195->171|195->171|196->172|199->175|199->175|200->176|200->176|200->176|201->177|203->179|203->179|205->181|234->210|234->210|235->211|236->212|236->212|237->213|246->222|246->222|247->223|249->225|249->225|251->227|251->227|252->228|254->230|254->230|255->231|261->237|261->237|262->238|262->238|262->238|263->239|264->240|264->240|265->241|266->242|266->242|268->244|270->246|270->246|271->247|272->248|272->248|277->253|277->253|278->254|279->255|279->255|280->256|286->262|286->262|289->265|289->265|291->267|292->268|292->268|293->269|294->270|294->270|295->271|302->278|302->278|305->281|305->281|307->283|308->284|308->284|310->286|311->287|311->287|312->288|315->291|315->291|318->294|318->294|320->296|321->297|321->297|322->298|323->299|323->299|324->300|331->307|331->307|332->308|332->308|333->309|333->309|334->310|334->310|336->312|336->312|336->312|337->313|343->319|343->319|344->320|345->321|345->321|347->323|350->326|350->326|351->327|353->329|353->329|354->330|355->331|355->331|356->332|358->334|358->334|360->336|360->336|360->336|361->337|361->337|361->337|362->338|365->341|365->341|366->342|367->343|367->343|368->344|370->346|370->346|371->347|371->347|373->349|375->351|375->351|377->353|377->353|377->353|378->354|403->379|403->379|404->380|405->381|405->381|406->382|408->384|408->384|408->384|408->384|408->384|409->385|411->387|411->387|412->388|417->393|417->393|418->394|418->394|420->396|421->397|421->397|422->398|423->399|423->399|424->400|424->400|424->400|425->401|426->402|426->402|427->403|427->403|427->403|428->404|429->405|429->405|429->405|429->405|429->405|430->406|431->407|431->407|432->408|433->409|433->409|433->409|433->409|433->409|434->410|434->410|434->410|435->411|436->412|436->412|436->412|436->412|436->412|437->413|438->414|438->414|439->415|440->416|440->416|441->417|441->417|442->418|442->418|442->418|443->419|444->420|444->420|445->421|446->422|446->422|446->422|446->422|446->422|447->423|448->424|448->424|449->425|449->425|451->427|452->428|452->428|453->429|453->429|453->429|454->430|457->433|457->433|458->434|458->434|462->438|463->439|463->439|464->440|464->440|464->440|465->441|468->444|468->444|469->445|470->446|470->446|471->447|472->448|472->448|472->448|472->448|472->448|473->449|474->450|474->450|475->451|475->451|475->451|475->451|475->451|476->452|479->455|479->455|481->457|486->462|486->462|487->463|487->463|490->466|490->466|491->467|491->467|491->467|492->468|497->473|497->473|498->474|503->479|503->479|504->480|504->480|505->481|505->481|507->483|507->483|508->484|511->487|511->487|512->488|512->488|512->488|513->489|515->491|515->491|516->492|517->493|517->493|518->494|518->494|519->495|521->497|521->497|522->498|522->498|523->499|523->499|524->500|524->500|526->502|528->504|528->504|529->505|529->505|529->505|530->506|532->508|532->508|533->509|533->509|534->510|534->510|535->511|535->511|535->511|536->512|538->514|538->514|539->515|539->515|540->516|540->516|541->517|541->517|541->517|542->518|544->520|544->520|545->521|545->521
                  -- GENERATED --
              */
          