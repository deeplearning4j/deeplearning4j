
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


Seq[Any](format.raw/*1.1*/("""<html>
    <head>
        <style type="text/css">
        /* Color and style reference.
            To change: do find + replace on comment + color

            heading background:         headingbgcol        #063E53            //Old candidates: #3B5998
            heading text color:         headingtextcolor    white

        */

        .hd """),format.raw/*12.13*/("""{"""),format.raw/*12.14*/("""
            """),format.raw/*13.13*/("""background-color: #000000;
            font-size: 18px;
            color: #FFFFFF;
        """),format.raw/*16.9*/("""}"""),format.raw/*16.10*/("""

        """),format.raw/*18.9*/("""html, body """),format.raw/*18.20*/("""{"""),format.raw/*18.21*/("""
            """),format.raw/*19.13*/("""width: 100%;
            height: 100%;
            padding: 0;
        """),format.raw/*22.9*/("""}"""),format.raw/*22.10*/("""

        """),format.raw/*24.9*/(""".bgcolor """),format.raw/*24.18*/("""{"""),format.raw/*24.19*/("""
            """),format.raw/*25.13*/("""background-color: #EFEFEF;
        """),format.raw/*26.9*/("""}"""),format.raw/*26.10*/("""

        """),format.raw/*28.9*/("""h1 """),format.raw/*28.12*/("""{"""),format.raw/*28.13*/("""
            """),format.raw/*29.13*/("""font-family: Georgia, Times, 'Times New Roman', serif;
            font-size: 28px;
            font-style: bold;
            font-variant: normal;
            font-weight: 500;
            line-height: 26.4px;
        """),format.raw/*35.9*/("""}"""),format.raw/*35.10*/("""

        """),format.raw/*37.9*/("""h3 """),format.raw/*37.12*/("""{"""),format.raw/*37.13*/("""
            """),format.raw/*38.13*/("""font-family: Georgia, Times, 'Times New Roman', serif;
            font-size: 16px;
            font-style: normal;
            font-variant: normal;
            font-weight: 500;
            line-height: 26.4px;
        """),format.raw/*44.9*/("""}"""),format.raw/*44.10*/("""

        """),format.raw/*46.9*/("""table.resultsTable """),format.raw/*46.28*/("""{"""),format.raw/*46.29*/("""
            """),format.raw/*47.13*/("""border-collapse:collapse;
            background-color: white;
            /*border-collapse: collapse;*/
            padding: 15px;
        """),format.raw/*51.9*/("""}"""),format.raw/*51.10*/("""

        """),format.raw/*53.9*/("""table.resultsTable td, table.resultsTable tr, table.resultsTable th """),format.raw/*53.77*/("""{"""),format.raw/*53.78*/("""
            """),format.raw/*54.13*/("""border:solid black 1px;
            white-space: pre;   /* assume text is preprocessed for formatting */
        """),format.raw/*56.9*/("""}"""),format.raw/*56.10*/("""

        """),format.raw/*58.9*/("""table.resultsTable th """),format.raw/*58.31*/("""{"""),format.raw/*58.32*/("""
            """),format.raw/*59.13*/("""background-color: /*headingbgcol*/#063E53;
            color: white;
            padding-left: 4px;
            padding-right: 4px;
        """),format.raw/*63.9*/("""}"""),format.raw/*63.10*/("""

        """),format.raw/*65.9*/("""table.resultsTable td """),format.raw/*65.31*/("""{"""),format.raw/*65.32*/("""
            """),format.raw/*66.13*/("""/*background-color: white;*/
            padding-left: 4px;
            padding-right: 4px;
        """),format.raw/*69.9*/("""}"""),format.raw/*69.10*/("""

        """),format.raw/*71.9*/("""/* Properties for table cells in the tables generated using the RenderableComponent mechanism */
        .renderableComponentTable """),format.raw/*72.35*/("""{"""),format.raw/*72.36*/("""
            """),format.raw/*73.13*/("""/*table-layout:fixed; */    /*Avoids scrollbar, but makes fixed width for all columns :( */
            width: 100%
        """),format.raw/*75.9*/("""}"""),format.raw/*75.10*/("""
        """),format.raw/*76.9*/(""".renderableComponentTable td """),format.raw/*76.38*/("""{"""),format.raw/*76.39*/("""
            """),format.raw/*77.13*/("""padding-left: 4px;
            padding-right: 4px;
            white-space: pre;   /* assume text is pre-processed (important for line breaks etc)*/
            word-wrap:break-word;
            vertical-align: top;
        """),format.raw/*82.9*/("""}"""),format.raw/*82.10*/("""

        """),format.raw/*84.9*/("""/** CSS for result table rows */
        .resultTableRow """),format.raw/*85.25*/("""{"""),format.raw/*85.26*/("""
            """),format.raw/*86.13*/("""background-color: #FFFFFF;
            cursor: pointer;
        """),format.raw/*88.9*/("""}"""),format.raw/*88.10*/("""

        """),format.raw/*90.9*/("""/** CSS for result table CONTENT rows (i.e., only visible when expanded) */
        .resultTableRowSelected """),format.raw/*91.33*/("""{"""),format.raw/*91.34*/("""
            """),format.raw/*92.13*/("""background-color: rgba(0, 157, 255, 0.16);
        """),format.raw/*93.9*/("""}"""),format.raw/*93.10*/("""

        """),format.raw/*95.9*/(""".resultsHeadingDiv """),format.raw/*95.28*/("""{"""),format.raw/*95.29*/("""
            """),format.raw/*96.13*/("""background-color: /*headingbgcol*/#063E53;
            color: white;
            font-family: Georgia, Times, 'Times New Roman', serif;
            font-size: 20px;
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
        """),format.raw/*112.9*/("""}"""),format.raw/*112.10*/("""

        """),format.raw/*114.9*/("""div.outerelements """),format.raw/*114.27*/("""{"""),format.raw/*114.28*/("""
            """),format.raw/*115.13*/("""padding-bottom: 30px;
        """),format.raw/*116.9*/("""}"""),format.raw/*116.10*/("""

        """),format.raw/*118.9*/("""#accordion, #accordion2 """),format.raw/*118.33*/("""{"""),format.raw/*118.34*/("""
            """),format.raw/*119.13*/("""padding-bottom: 20px;
        """),format.raw/*120.9*/("""}"""),format.raw/*120.10*/("""

        """),format.raw/*122.9*/("""#accordion .ui-accordion-header, #accordion2 .ui-accordion-header, #accordion3 .ui-accordion-header """),format.raw/*122.109*/("""{"""),format.raw/*122.110*/("""
            """),format.raw/*123.13*/("""background-color: /*headingbgcolor*/#063E53;      /*Color when collapsed*/
            color: /*headingtextcolor*/white;
            font-family: Georgia, Times, 'Times New Roman', serif;
            font-size: 20px;
            font-style: bold;
            font-variant: normal;
            margin: 0px;
            background-image: none;     /* Necessary, otherwise color changes don't make a difference */
        """),format.raw/*131.9*/("""}"""),format.raw/*131.10*/("""

        """),format.raw/*133.9*/("""#accordion .ui-accordion-content """),format.raw/*133.42*/("""{"""),format.raw/*133.43*/("""
            """),format.raw/*134.13*/("""width: 100%;
            background-color: white;    /*background color of accordian content (elements in front may have different color */
            color: black;  /* text etc color */
            font-size: 10pt;
            line-height: 16pt;
            overflow:visible !important;
        """),format.raw/*140.9*/("""}"""),format.raw/*140.10*/("""

        """),format.raw/*142.9*/("""/** Line charts */
        path """),format.raw/*143.14*/("""{"""),format.raw/*143.15*/("""
            """),format.raw/*144.13*/("""stroke: steelblue;
            stroke-width: 2;
            fill: none;
        """),format.raw/*147.9*/("""}"""),format.raw/*147.10*/("""
        """),format.raw/*148.9*/(""".axis path, .axis line """),format.raw/*148.32*/("""{"""),format.raw/*148.33*/("""
            """),format.raw/*149.13*/("""fill: none;
            stroke: #000;
            shape-rendering: crispEdges;
        """),format.raw/*152.9*/("""}"""),format.raw/*152.10*/("""
        """),format.raw/*153.9*/(""".tick line """),format.raw/*153.20*/("""{"""),format.raw/*153.21*/("""
            """),format.raw/*154.13*/("""opacity: 0.2;
            shape-rendering: crispEdges;
        """),format.raw/*156.9*/("""}"""),format.raw/*156.10*/("""

        """),format.raw/*158.9*/("""</style>
        <title>Arbiter UI</title>
    </head>
    <body class="bgcolor">

        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="stylesheet" href="/assets/css/arbiter/bootstrap.min.css">

        """),format.raw/*166.91*/("""
        """),format.raw/*167.9*/("""<script src="/assets/js/jquery-1.9.1.min.js"></script>
        <link rel="stylesheet" href="/assets/css/arbiter/jquery-ui.css">
        <script src="/assets/js/arbiter/jquery-1.10.2.js"></script>
        <script src="/assets/js/arbiter/jquery-ui.js"></script>
        <script src="/assets/js/arbiter/d3.min.js"></script>
        <script src="/assets/js/arbiter/bootstrap.min.js"></script>
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
    setInterval(function()"""),format.raw/*188.27*/("""{"""),format.raw/*188.28*/("""
        """),format.raw/*189.9*/("""//Get the update status, and do something with it:
        $.get("/arbiter/lastUpdate",function(data)"""),format.raw/*190.51*/("""{"""),format.raw/*190.52*/("""
            """),format.raw/*191.13*/("""//Encoding: matches names in UpdateStatus class
            var jsonObj = JSON.parse(JSON.stringify(data));
            var statusTime = jsonObj['statusUpdateTime'];
            var settingsTime = jsonObj['settingsUpdateTime'];
            var resultsTime = jsonObj['resultsUpdateTime'];
            //console.log("Last update times: " + statusTime + ", " + settingsTime + ", " + resultsTime);

            //Check last update times for each part of document, and update as necessary
            //First section: summary status
            if(lastStatusUpdateTime != statusTime)"""),format.raw/*200.51*/("""{"""),format.raw/*200.52*/("""
                """),format.raw/*201.17*/("""//Get JSON: address set by SummaryStatusResource
                $.get("/arbiter/summary",function(data)"""),format.raw/*202.56*/("""{"""),format.raw/*202.57*/("""
                    """),format.raw/*203.21*/("""var summaryStatusDiv = $('#statusdiv');
                    summaryStatusDiv.html('');

                    var str = JSON.stringify(data);
                    var component = Component.getComponent(str);
                    component.render(summaryStatusDiv);
                """),format.raw/*209.17*/("""}"""),format.raw/*209.18*/(""");

                lastStatusUpdateTime = statusTime;
            """),format.raw/*212.13*/("""}"""),format.raw/*212.14*/("""

            """),format.raw/*214.13*/("""//Second section: Optimization settings
            if(lastSettingsUpdateTime != settingsTime)"""),format.raw/*215.55*/("""{"""),format.raw/*215.56*/("""
                """),format.raw/*216.17*/("""//Get JSON for components
                $.get("/arbiter/config",function(data)"""),format.raw/*217.55*/("""{"""),format.raw/*217.56*/("""
                    """),format.raw/*218.21*/("""var str = JSON.stringify(data);

                    var configDiv = $('#settingsdiv');
                    configDiv.html('');

                    var component = Component.getComponent(str);
                    component.render(configDiv);
                """),format.raw/*225.17*/("""}"""),format.raw/*225.18*/(""");

                lastSettingsUpdateTime = settingsTime;
            """),format.raw/*228.13*/("""}"""),format.raw/*228.14*/("""

            """),format.raw/*230.13*/("""//Third section: Summary results table (summary info for each candidate)
            if(lastResultsUpdateTime != resultsTime)"""),format.raw/*231.53*/("""{"""),format.raw/*231.54*/("""

                """),format.raw/*233.17*/("""//Get JSON; address set by SummaryResultsResource
                $.get("/arbiter/results",function(data)"""),format.raw/*234.56*/("""{"""),format.raw/*234.57*/("""
                    """),format.raw/*235.21*/("""//Expect an array of CandidateInfo type objects here
                    resultsTableContent = data;
                    drawResultTable();
                """),format.raw/*238.17*/("""}"""),format.raw/*238.18*/(""");

                lastResultsUpdateTime = resultsTime;
            """),format.raw/*241.13*/("""}"""),format.raw/*241.14*/("""

            """),format.raw/*243.13*/("""//Finally: Currently selected result
            if(selectedCandidateIdx != null)"""),format.raw/*244.45*/("""{"""),format.raw/*244.46*/("""
                """),format.raw/*245.17*/("""//Get JSON for components
                $.get("/arbiter/candidateInfo/"+selectedCandidateIdx,function(data)"""),format.raw/*246.84*/("""{"""),format.raw/*246.85*/("""
                    """),format.raw/*247.21*/("""var str = JSON.stringify(data);

                    var resultsViewDiv = $('#resultsviewdiv');
                    resultsViewDiv.html('');

                    var component = Component.getComponent(str);
                    component.render(resultsViewDiv);
                """),format.raw/*254.17*/("""}"""),format.raw/*254.18*/(""");
            """),format.raw/*255.13*/("""}"""),format.raw/*255.14*/("""
        """),format.raw/*256.9*/("""}"""),format.raw/*256.10*/(""")
    """),format.raw/*257.5*/("""}"""),format.raw/*257.6*/(""",2000);    //Loop every 2 seconds

    function createTable(tableObj,tableId,appendTo)"""),format.raw/*259.52*/("""{"""),format.raw/*259.53*/("""
        """),format.raw/*260.9*/("""//Expect RenderableComponentTable
        var header = tableObj['header'];
        var values = tableObj['table'];
        var title = tableObj['title'];
        var nRows = (values ? values.length : 0);

        if(title)"""),format.raw/*266.18*/("""{"""),format.raw/*266.19*/("""
            """),format.raw/*267.13*/("""appendTo.append("<h5>"+title+"</h5>");
        """),format.raw/*268.9*/("""}"""),format.raw/*268.10*/("""

        """),format.raw/*270.9*/("""var table;
        if(tableId) table = $("<table id=\"" + tableId + "\" class=\"renderableComponentTable\">");
        else table = $("<table class=\"renderableComponentTable\">");
        if(header)"""),format.raw/*273.19*/("""{"""),format.raw/*273.20*/("""
            """),format.raw/*274.13*/("""var headerRow = $("<tr>");
            var len = header.length;
            for( var i=0; i<len; i++ )"""),format.raw/*276.39*/("""{"""),format.raw/*276.40*/("""
                """),format.raw/*277.17*/("""headerRow.append($("<th>" + header[i] + "</th>"));
            """),format.raw/*278.13*/("""}"""),format.raw/*278.14*/("""
            """),format.raw/*279.13*/("""headerRow.append($("</tr>"));
            table.append(headerRow);
        """),format.raw/*281.9*/("""}"""),format.raw/*281.10*/("""

        """),format.raw/*283.9*/("""if(values)"""),format.raw/*283.19*/("""{"""),format.raw/*283.20*/("""
            """),format.raw/*284.13*/("""for( var i=0; i<nRows; i++ )"""),format.raw/*284.41*/("""{"""),format.raw/*284.42*/("""
                """),format.raw/*285.17*/("""var row = $("<tr>");
                var rowValues = values[i];
                var len = rowValues.length;
                for( var j=0; j<len; j++ )"""),format.raw/*288.43*/("""{"""),format.raw/*288.44*/("""
                    """),format.raw/*289.21*/("""row.append($('<td>'+rowValues[j]+'</td>'));
                """),format.raw/*290.17*/("""}"""),format.raw/*290.18*/("""
                """),format.raw/*291.17*/("""row.append($("</tr>"));
                table.append(row);
            """),format.raw/*293.13*/("""}"""),format.raw/*293.14*/("""
        """),format.raw/*294.9*/("""}"""),format.raw/*294.10*/("""

        """),format.raw/*296.9*/("""table.append($("</table>"));
        appendTo.append(table);
    """),format.raw/*298.5*/("""}"""),format.raw/*298.6*/("""

    """),format.raw/*300.5*/("""function drawResultTable()"""),format.raw/*300.31*/("""{"""),format.raw/*300.32*/("""

        """),format.raw/*302.9*/("""//Remove all elements from the table body
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
        for(var i=0; i<len; i++)"""),format.raw/*327.33*/("""{"""),format.raw/*327.34*/("""
"""),format.raw/*328.1*/("""//            var row = $('<tr class="resultTableRow" id="rTbl-' + sorted[i][0] + '"/>');
            var row;
            if(selectedCandidateIdx == sorted[i][0])"""),format.raw/*330.53*/("""{"""),format.raw/*330.54*/("""
                """),format.raw/*331.17*/("""//Selected row
                row = $('<tr class="resultTableRowSelected" id="rTbl-' + sorted[i][0] + '"/>');
            """),format.raw/*333.13*/("""}"""),format.raw/*333.14*/(""" """),format.raw/*333.15*/("""else """),format.raw/*333.20*/("""{"""),format.raw/*333.21*/("""
                """),format.raw/*334.17*/("""//Normal row
                row = $('<tr class="resultTableRow" id="rTbl-' + sorted[i][0] + '"/>');
            """),format.raw/*336.13*/("""}"""),format.raw/*336.14*/("""
            """),format.raw/*337.13*/("""row.append($("<td>" + sorted[i][0] + "</td>"));
            var score = sorted[i][1];
            row.append($("<td>" + ((!score || score == "null") ? "-" : score) + "</td>"));
            row.append($("<td>" + sorted[i][2] + "</td>"));
            tableBody.append(row);
        """),format.raw/*342.9*/("""}"""),format.raw/*342.10*/("""
    """),format.raw/*343.5*/("""}"""),format.raw/*343.6*/("""

    """),format.raw/*345.5*/("""//Compare function for results, based on sort order
    function compareResultsIndex(a, b)"""),format.raw/*346.39*/("""{"""),format.raw/*346.40*/("""
        """),format.raw/*347.9*/("""return (resultTableSortOrder == "ascending" ? a[0] - b[0] : b[0] - a[0]);
    """),format.raw/*348.5*/("""}"""),format.raw/*348.6*/("""
    """),format.raw/*349.5*/("""function compareScores(a,b)"""),format.raw/*349.32*/("""{"""),format.raw/*349.33*/("""
        """),format.raw/*350.9*/("""//TODO Not always numbers...
        if(resultTableSortOrder == "ascending")"""),format.raw/*351.48*/("""{"""),format.raw/*351.49*/("""
            """),format.raw/*352.13*/("""return a[1] - b[1];
        """),format.raw/*353.9*/("""}"""),format.raw/*353.10*/(""" """),format.raw/*353.11*/("""else """),format.raw/*353.16*/("""{"""),format.raw/*353.17*/("""
            """),format.raw/*354.13*/("""return b[1] - a[1];
        """),format.raw/*355.9*/("""}"""),format.raw/*355.10*/("""
    """),format.raw/*356.5*/("""}"""),format.raw/*356.6*/("""
    """),format.raw/*357.5*/("""function compareStatus(a,b)"""),format.raw/*357.32*/("""{"""),format.raw/*357.33*/("""
        """),format.raw/*358.9*/("""//TODO: secondary sort on... score? index?
        if(resultTableSortOrder == "ascending")"""),format.raw/*359.48*/("""{"""),format.raw/*359.49*/("""
            """),format.raw/*360.13*/("""return (a[2] < b[2] ? -1 : (a[2] > b[2] ? 1 : 0));
        """),format.raw/*361.9*/("""}"""),format.raw/*361.10*/(""" """),format.raw/*361.11*/("""else """),format.raw/*361.16*/("""{"""),format.raw/*361.17*/("""
            """),format.raw/*362.13*/("""return (a[2] < b[2] ? 1 : (a[2] > b[2] ? -1 : 0));
        """),format.raw/*363.9*/("""}"""),format.raw/*363.10*/("""
    """),format.raw/*364.5*/("""}"""),format.raw/*364.6*/("""

    """),format.raw/*366.5*/("""//Do a HTTP request on the specified path, parse and insert into the provided element
    function loadCandidateDetails(path, elementToAppendTo)"""),format.raw/*367.59*/("""{"""),format.raw/*367.60*/("""
        """),format.raw/*368.9*/("""$.get(path, function (data) """),format.raw/*368.37*/("""{"""),format.raw/*368.38*/("""
            """),format.raw/*369.13*/("""var str = JSON.stringify(data);
            var component = Component.getComponent(str);
            component.render(elementToAppendTo);
        """),format.raw/*372.9*/("""}"""),format.raw/*372.10*/(""");
    """),format.raw/*373.5*/("""}"""),format.raw/*373.6*/("""



    """),format.raw/*377.5*/("""//Sorting by column: Intercept click events on table header
    $(function()"""),format.raw/*378.17*/("""{"""),format.raw/*378.18*/("""
        """),format.raw/*379.9*/("""$("#resultsTableHeader").delegate("th", "click", function(e) """),format.raw/*379.70*/("""{"""),format.raw/*379.71*/("""
            """),format.raw/*380.13*/("""//console.log("Header clicked on at: " + $(e.currentTarget).index() + " - " + $(e.currentTarget).html());
            //Update the sort order for the table:
            var clickIndex = $(e.currentTarget).index();
            if(clickIndex == resultTableSortIndex)"""),format.raw/*383.51*/("""{"""),format.raw/*383.52*/("""
                """),format.raw/*384.17*/("""//Switch sort order: ascending -> descending or descending -> ascending
                if(resultTableSortOrder == "ascending")"""),format.raw/*385.56*/("""{"""),format.raw/*385.57*/("""
                    """),format.raw/*386.21*/("""resultTableSortOrder = "descending";
                """),format.raw/*387.17*/("""}"""),format.raw/*387.18*/(""" """),format.raw/*387.19*/("""else """),format.raw/*387.24*/("""{"""),format.raw/*387.25*/("""
                    """),format.raw/*388.21*/("""resultTableSortOrder = "ascending";
                """),format.raw/*389.17*/("""}"""),format.raw/*389.18*/("""
            """),format.raw/*390.13*/("""}"""),format.raw/*390.14*/(""" """),format.raw/*390.15*/("""else """),format.raw/*390.20*/("""{"""),format.raw/*390.21*/("""
                """),format.raw/*391.17*/("""//Sort on column, ascending:
                resultTableSortIndex = clickIndex;
                resultTableSortOrder = "ascending";
            """),format.raw/*394.13*/("""}"""),format.raw/*394.14*/("""

            """),format.raw/*396.13*/("""//Clear record of expanded rows
            expandedRowsCandidateIDs = [];

            //Redraw table
            drawResultTable();
        """),format.raw/*401.9*/("""}"""),format.raw/*401.10*/(""");
    """),format.raw/*402.5*/("""}"""),format.raw/*402.6*/(""");

    //Displaying model/candidate details: Intercept click events on table rows -> toggle visibility on content rows
    $(function()"""),format.raw/*405.17*/("""{"""),format.raw/*405.18*/("""
        """),format.raw/*406.9*/("""$("#resultsTableBody").delegate("tr", "click", function(e)"""),format.raw/*406.67*/("""{"""),format.raw/*406.68*/("""
            """),format.raw/*407.13*/("""var id = this.id;   //Expect: rTbl-X  where X is some index
            var dashIdx = id.indexOf("-");
            var candidateID = Number(id.substring(dashIdx+1));

//            console.log("Clicked row: " + this.id + " with class: " + this.className + ", candidateId = " + candidateID);

            if(this.className == "resultTableRow")"""),format.raw/*413.51*/("""{"""),format.raw/*413.52*/("""
                """),format.raw/*414.17*/("""//Set selected model
                selectedCandidateIdx = candidateID;

                //TODO fire off update
            """),format.raw/*418.13*/("""}"""),format.raw/*418.14*/("""
        """),format.raw/*419.9*/("""}"""),format.raw/*419.10*/(""");
    """),format.raw/*420.5*/("""}"""),format.raw/*420.6*/(""");

</script>
<script>
    $(function () """),format.raw/*424.19*/("""{"""),format.raw/*424.20*/("""
        """),format.raw/*425.9*/("""$("#accordion").accordion("""),format.raw/*425.35*/("""{"""),format.raw/*425.36*/("""
            """),format.raw/*426.13*/("""collapsible: true,
            heightStyle: "content"
        """),format.raw/*428.9*/("""}"""),format.raw/*428.10*/(""");
    """),format.raw/*429.5*/("""}"""),format.raw/*429.6*/(""");
    $(function () """),format.raw/*430.19*/("""{"""),format.raw/*430.20*/("""
        """),format.raw/*431.9*/("""$("#accordion2").accordion("""),format.raw/*431.36*/("""{"""),format.raw/*431.37*/("""
            """),format.raw/*432.13*/("""collapsible: true,
            heightStyle: "content"
        """),format.raw/*434.9*/("""}"""),format.raw/*434.10*/(""");
    """),format.raw/*435.5*/("""}"""),format.raw/*435.6*/(""");
    $(function () """),format.raw/*436.19*/("""{"""),format.raw/*436.20*/("""
        """),format.raw/*437.9*/("""$("#accordion3").accordion("""),format.raw/*437.36*/("""{"""),format.raw/*437.37*/("""
            """),format.raw/*438.13*/("""collapsible: true,
            heightStyle: "content"
        """),format.raw/*440.9*/("""}"""),format.raw/*440.10*/(""");
    """),format.raw/*441.5*/("""}"""),format.raw/*441.6*/(""");
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
                            margin-top: 12px">Arbiter UI</div></td>
                </tr>
            </tbody>
        </table>

        <div style="width: 1400px;
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
                  DATE: Fri Jul 21 14:08:59 AEST 2017
                  SOURCE: C:/DL4J/Git/Arbiter/arbiter-ui/src/main/views/org/deeplearning4j/arbiter/ui/views/ArbiterUI.scala.html
                  HASH: e71aafb7bf98f65dd179324c4e69609718d4d1d5
                  MATRIX: 647->0|1031->356|1060->357|1102->371|1224->466|1253->467|1292->479|1331->490|1360->491|1402->505|1503->579|1532->580|1571->592|1608->601|1637->602|1679->616|1742->652|1771->653|1810->665|1841->668|1870->669|1912->683|2164->908|2193->909|2232->921|2263->924|2292->925|2334->939|2588->1166|2617->1167|2656->1179|2703->1198|2732->1199|2774->1213|2946->1358|2975->1359|3014->1371|3110->1439|3139->1440|3181->1454|3323->1569|3352->1570|3391->1582|3441->1604|3470->1605|3512->1619|3683->1763|3712->1764|3751->1776|3801->1798|3830->1799|3872->1813|4002->1916|4031->1917|4070->1929|4230->2061|4259->2062|4301->2076|4454->2202|4483->2203|4520->2213|4577->2242|4606->2243|4648->2257|4904->2486|4933->2487|4972->2499|5058->2557|5087->2558|5129->2572|5222->2638|5251->2639|5290->2651|5427->2760|5456->2761|5498->2775|5577->2827|5606->2828|5645->2840|5692->2859|5721->2860|5763->2874|6363->3446|6393->3447|6433->3459|6480->3477|6510->3478|6553->3492|6612->3523|6642->3524|6682->3536|6735->3560|6765->3561|6808->3575|6867->3606|6897->3607|6937->3619|7067->3719|7098->3720|7141->3734|7596->4161|7626->4162|7666->4174|7728->4207|7758->4208|7801->4222|8132->4525|8162->4526|8202->4538|8264->4571|8294->4572|8337->4586|8448->4669|8478->4670|8516->4680|8568->4703|8598->4704|8641->4718|8759->4808|8789->4809|8827->4819|8867->4830|8897->4831|8940->4845|9033->4910|9063->4911|9103->4923|9386->5259|9424->5269|10292->6108|10322->6109|10360->6119|10491->6221|10521->6222|10564->6236|11180->6823|11210->6824|11257->6842|11391->6947|11421->6948|11472->6970|11784->7253|11814->7254|11913->7324|11943->7325|11988->7341|12112->7436|12142->7437|12189->7455|12299->7536|12329->7537|12380->7559|12675->7825|12705->7826|12808->7900|12838->7901|12883->7917|13038->8043|13068->8044|13117->8064|13252->8170|13282->8171|13333->8193|13521->8352|13551->8353|13652->8425|13682->8426|13727->8442|13838->8524|13868->8525|13915->8543|14054->8653|14084->8654|14135->8676|14448->8960|14478->8961|14523->8977|14553->8978|14591->8988|14621->8989|14656->8996|14685->8997|14802->9085|14832->9086|14870->9096|15127->9324|15157->9325|15200->9339|15276->9387|15306->9388|15346->9400|15577->9602|15607->9603|15650->9617|15783->9721|15813->9722|15860->9740|15953->9804|15983->9805|16026->9819|16131->9896|16161->9897|16201->9909|16240->9919|16270->9920|16313->9934|16370->9962|16400->9963|16447->9981|16629->10134|16659->10135|16710->10157|16800->10218|16830->10219|16877->10237|16979->10310|17009->10311|17047->10321|17077->10322|17117->10334|17212->10401|17241->10402|17277->10410|17332->10436|17362->10437|17402->10449|18759->11777|18789->11778|18819->11780|19013->11945|19043->11946|19090->11964|19244->12089|19274->12090|19304->12091|19338->12096|19368->12097|19415->12115|19559->12230|19589->12231|19632->12245|19945->12530|19975->12531|20009->12537|20038->12538|20074->12546|20194->12637|20224->12638|20262->12648|20369->12727|20398->12728|20432->12734|20488->12761|20518->12762|20556->12772|20662->12849|20692->12850|20735->12864|20792->12893|20822->12894|20852->12895|20886->12900|20916->12901|20959->12915|21016->12944|21046->12945|21080->12951|21109->12952|21143->12958|21199->12985|21229->12986|21267->12996|21387->13087|21417->13088|21460->13102|21548->13162|21578->13163|21608->13164|21642->13169|21672->13170|21715->13184|21803->13244|21833->13245|21867->13251|21896->13252|21932->13260|22106->13405|22136->13406|22174->13416|22231->13444|22261->13445|22304->13459|22481->13608|22511->13609|22547->13617|22576->13618|22616->13630|22722->13707|22752->13708|22790->13718|22880->13779|22910->13780|22953->13794|23249->14061|23279->14062|23326->14080|23483->14208|23513->14209|23564->14231|23647->14285|23677->14286|23707->14287|23741->14292|23771->14293|23822->14315|23904->14368|23934->14369|23977->14383|24007->14384|24037->14385|24071->14390|24101->14391|24148->14409|24324->14556|24354->14557|24399->14573|24574->14720|24604->14721|24640->14729|24669->14730|24837->14869|24867->14870|24905->14880|24992->14938|25022->14939|25065->14953|25442->15301|25472->15302|25519->15320|25677->15449|25707->15450|25745->15460|25775->15461|25811->15469|25840->15470|25914->15515|25944->15516|25982->15526|26037->15552|26067->15553|26110->15567|26202->15631|26232->15632|26268->15640|26297->15641|26348->15663|26378->15664|26416->15674|26472->15701|26502->15702|26545->15716|26637->15780|26667->15781|26703->15789|26732->15790|26783->15812|26813->15813|26851->15823|26907->15850|26937->15851|26980->15865|27072->15929|27102->15930|27138->15938|27167->15939
                  LINES: 25->1|36->12|36->12|37->13|40->16|40->16|42->18|42->18|42->18|43->19|46->22|46->22|48->24|48->24|48->24|49->25|50->26|50->26|52->28|52->28|52->28|53->29|59->35|59->35|61->37|61->37|61->37|62->38|68->44|68->44|70->46|70->46|70->46|71->47|75->51|75->51|77->53|77->53|77->53|78->54|80->56|80->56|82->58|82->58|82->58|83->59|87->63|87->63|89->65|89->65|89->65|90->66|93->69|93->69|95->71|96->72|96->72|97->73|99->75|99->75|100->76|100->76|100->76|101->77|106->82|106->82|108->84|109->85|109->85|110->86|112->88|112->88|114->90|115->91|115->91|116->92|117->93|117->93|119->95|119->95|119->95|120->96|136->112|136->112|138->114|138->114|138->114|139->115|140->116|140->116|142->118|142->118|142->118|143->119|144->120|144->120|146->122|146->122|146->122|147->123|155->131|155->131|157->133|157->133|157->133|158->134|164->140|164->140|166->142|167->143|167->143|168->144|171->147|171->147|172->148|172->148|172->148|173->149|176->152|176->152|177->153|177->153|177->153|178->154|180->156|180->156|182->158|190->166|191->167|212->188|212->188|213->189|214->190|214->190|215->191|224->200|224->200|225->201|226->202|226->202|227->203|233->209|233->209|236->212|236->212|238->214|239->215|239->215|240->216|241->217|241->217|242->218|249->225|249->225|252->228|252->228|254->230|255->231|255->231|257->233|258->234|258->234|259->235|262->238|262->238|265->241|265->241|267->243|268->244|268->244|269->245|270->246|270->246|271->247|278->254|278->254|279->255|279->255|280->256|280->256|281->257|281->257|283->259|283->259|284->260|290->266|290->266|291->267|292->268|292->268|294->270|297->273|297->273|298->274|300->276|300->276|301->277|302->278|302->278|303->279|305->281|305->281|307->283|307->283|307->283|308->284|308->284|308->284|309->285|312->288|312->288|313->289|314->290|314->290|315->291|317->293|317->293|318->294|318->294|320->296|322->298|322->298|324->300|324->300|324->300|326->302|351->327|351->327|352->328|354->330|354->330|355->331|357->333|357->333|357->333|357->333|357->333|358->334|360->336|360->336|361->337|366->342|366->342|367->343|367->343|369->345|370->346|370->346|371->347|372->348|372->348|373->349|373->349|373->349|374->350|375->351|375->351|376->352|377->353|377->353|377->353|377->353|377->353|378->354|379->355|379->355|380->356|380->356|381->357|381->357|381->357|382->358|383->359|383->359|384->360|385->361|385->361|385->361|385->361|385->361|386->362|387->363|387->363|388->364|388->364|390->366|391->367|391->367|392->368|392->368|392->368|393->369|396->372|396->372|397->373|397->373|401->377|402->378|402->378|403->379|403->379|403->379|404->380|407->383|407->383|408->384|409->385|409->385|410->386|411->387|411->387|411->387|411->387|411->387|412->388|413->389|413->389|414->390|414->390|414->390|414->390|414->390|415->391|418->394|418->394|420->396|425->401|425->401|426->402|426->402|429->405|429->405|430->406|430->406|430->406|431->407|437->413|437->413|438->414|442->418|442->418|443->419|443->419|444->420|444->420|448->424|448->424|449->425|449->425|449->425|450->426|452->428|452->428|453->429|453->429|454->430|454->430|455->431|455->431|455->431|456->432|458->434|458->434|459->435|459->435|460->436|460->436|461->437|461->437|461->437|462->438|464->440|464->440|465->441|465->441
                  -- GENERATED --
              */
          