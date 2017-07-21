
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
    setInterval(doUpdate,5000);    //Loop every 5 seconds


    function doUpdate()"""),format.raw/*191.24*/("""{"""),format.raw/*191.25*/("""
        """),format.raw/*192.9*/("""//Get the update status, and do something with it:
        $.get("/arbiter/lastUpdate",function(data)"""),format.raw/*193.51*/("""{"""),format.raw/*193.52*/("""
            """),format.raw/*194.13*/("""//Encoding: matches names in UpdateStatus class
            var jsonObj = JSON.parse(JSON.stringify(data));
            var statusTime = jsonObj['statusUpdateTime'];
            var settingsTime = jsonObj['settingsUpdateTime'];
            var resultsTime = jsonObj['resultsUpdateTime'];
            //console.log("Last update times: " + statusTime + ", " + settingsTime + ", " + resultsTime);

            //Check last update times for each part of document, and update as necessary
            //First section: summary status
            if(lastStatusUpdateTime != statusTime)"""),format.raw/*203.51*/("""{"""),format.raw/*203.52*/("""
                """),format.raw/*204.17*/("""//Get JSON: address set by SummaryStatusResource
                $.get("/arbiter/summary",function(data)"""),format.raw/*205.56*/("""{"""),format.raw/*205.57*/("""
                    """),format.raw/*206.21*/("""var summaryStatusDiv = $('#statusdiv');
                    summaryStatusDiv.html('');

                    var str = JSON.stringify(data);
                    var component = Component.getComponent(str);
                    component.render(summaryStatusDiv);
                """),format.raw/*212.17*/("""}"""),format.raw/*212.18*/(""");

                lastStatusUpdateTime = statusTime;
            """),format.raw/*215.13*/("""}"""),format.raw/*215.14*/("""

            """),format.raw/*217.13*/("""//Second section: Optimization settings
            if(lastSettingsUpdateTime != settingsTime)"""),format.raw/*218.55*/("""{"""),format.raw/*218.56*/("""
                """),format.raw/*219.17*/("""//Get JSON for components
                $.get("/arbiter/config",function(data)"""),format.raw/*220.55*/("""{"""),format.raw/*220.56*/("""
                    """),format.raw/*221.21*/("""var str = JSON.stringify(data);

                    var configDiv = $('#settingsdiv');
                    configDiv.html('');

                    var component = Component.getComponent(str);
                    component.render(configDiv);
                """),format.raw/*228.17*/("""}"""),format.raw/*228.18*/(""");

                lastSettingsUpdateTime = settingsTime;
            """),format.raw/*231.13*/("""}"""),format.raw/*231.14*/("""

            """),format.raw/*233.13*/("""//Third section: Summary results table (summary info for each candidate)
            if(lastResultsUpdateTime != resultsTime)"""),format.raw/*234.53*/("""{"""),format.raw/*234.54*/("""

                """),format.raw/*236.17*/("""//Get JSON; address set by SummaryResultsResource
                $.get("/arbiter/results",function(data)"""),format.raw/*237.56*/("""{"""),format.raw/*237.57*/("""
                    """),format.raw/*238.21*/("""//Expect an array of CandidateInfo type objects here
                    resultsTableContent = data;
                    drawResultTable();
                """),format.raw/*241.17*/("""}"""),format.raw/*241.18*/(""");

                lastResultsUpdateTime = resultsTime;
            """),format.raw/*244.13*/("""}"""),format.raw/*244.14*/("""

            """),format.raw/*246.13*/("""//Finally: Currently selected result
            if(selectedCandidateIdx != null)"""),format.raw/*247.45*/("""{"""),format.raw/*247.46*/("""
                """),format.raw/*248.17*/("""//Get JSON for components
                $.get("/arbiter/candidateInfo/"+selectedCandidateIdx,function(data)"""),format.raw/*249.84*/("""{"""),format.raw/*249.85*/("""
                    """),format.raw/*250.21*/("""var str = JSON.stringify(data);

                    var resultsViewDiv = $('#resultsviewdiv');
                    resultsViewDiv.html('');

                    var component = Component.getComponent(str);
                    component.render(resultsViewDiv);
                """),format.raw/*257.17*/("""}"""),format.raw/*257.18*/(""");
            """),format.raw/*258.13*/("""}"""),format.raw/*258.14*/("""
        """),format.raw/*259.9*/("""}"""),format.raw/*259.10*/(""")
    """),format.raw/*260.5*/("""}"""),format.raw/*260.6*/("""

    """),format.raw/*262.5*/("""function createTable(tableObj,tableId,appendTo)"""),format.raw/*262.52*/("""{"""),format.raw/*262.53*/("""
        """),format.raw/*263.9*/("""//Expect RenderableComponentTable
        var header = tableObj['header'];
        var values = tableObj['table'];
        var title = tableObj['title'];
        var nRows = (values ? values.length : 0);

        if(title)"""),format.raw/*269.18*/("""{"""),format.raw/*269.19*/("""
            """),format.raw/*270.13*/("""appendTo.append("<h5>"+title+"</h5>");
        """),format.raw/*271.9*/("""}"""),format.raw/*271.10*/("""

        """),format.raw/*273.9*/("""var table;
        if(tableId) table = $("<table id=\"" + tableId + "\" class=\"renderableComponentTable\">");
        else table = $("<table class=\"renderableComponentTable\">");
        if(header)"""),format.raw/*276.19*/("""{"""),format.raw/*276.20*/("""
            """),format.raw/*277.13*/("""var headerRow = $("<tr>");
            var len = header.length;
            for( var i=0; i<len; i++ )"""),format.raw/*279.39*/("""{"""),format.raw/*279.40*/("""
                """),format.raw/*280.17*/("""headerRow.append($("<th>" + header[i] + "</th>"));
            """),format.raw/*281.13*/("""}"""),format.raw/*281.14*/("""
            """),format.raw/*282.13*/("""headerRow.append($("</tr>"));
            table.append(headerRow);
        """),format.raw/*284.9*/("""}"""),format.raw/*284.10*/("""

        """),format.raw/*286.9*/("""if(values)"""),format.raw/*286.19*/("""{"""),format.raw/*286.20*/("""
            """),format.raw/*287.13*/("""for( var i=0; i<nRows; i++ )"""),format.raw/*287.41*/("""{"""),format.raw/*287.42*/("""
                """),format.raw/*288.17*/("""var row = $("<tr>");
                var rowValues = values[i];
                var len = rowValues.length;
                for( var j=0; j<len; j++ )"""),format.raw/*291.43*/("""{"""),format.raw/*291.44*/("""
                    """),format.raw/*292.21*/("""row.append($('<td>'+rowValues[j]+'</td>'));
                """),format.raw/*293.17*/("""}"""),format.raw/*293.18*/("""
                """),format.raw/*294.17*/("""row.append($("</tr>"));
                table.append(row);
            """),format.raw/*296.13*/("""}"""),format.raw/*296.14*/("""
        """),format.raw/*297.9*/("""}"""),format.raw/*297.10*/("""

        """),format.raw/*299.9*/("""table.append($("</table>"));
        appendTo.append(table);
    """),format.raw/*301.5*/("""}"""),format.raw/*301.6*/("""

    """),format.raw/*303.5*/("""function drawResultTable()"""),format.raw/*303.31*/("""{"""),format.raw/*303.32*/("""

        """),format.raw/*305.9*/("""//Remove all elements from the table body
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
        for(var i=0; i<len; i++)"""),format.raw/*330.33*/("""{"""),format.raw/*330.34*/("""
"""),format.raw/*331.1*/("""//            var row = $('<tr class="resultTableRow" id="rTbl-' + sorted[i][0] + '"/>');
            var row;
            if(selectedCandidateIdx == sorted[i][0])"""),format.raw/*333.53*/("""{"""),format.raw/*333.54*/("""
                """),format.raw/*334.17*/("""//Selected row
                row = $('<tr class="resultTableRowSelected" id="rTbl-' + sorted[i][0] + '"/>');
            """),format.raw/*336.13*/("""}"""),format.raw/*336.14*/(""" """),format.raw/*336.15*/("""else """),format.raw/*336.20*/("""{"""),format.raw/*336.21*/("""
                """),format.raw/*337.17*/("""//Normal row
                row = $('<tr class="resultTableRow" id="rTbl-' + sorted[i][0] + '"/>');
            """),format.raw/*339.13*/("""}"""),format.raw/*339.14*/("""
            """),format.raw/*340.13*/("""row.append($("<td>" + sorted[i][0] + "</td>"));
            var score = sorted[i][1];
            row.append($("<td>" + ((!score || score == "null") ? "-" : score) + "</td>"));
            row.append($("<td>" + sorted[i][2] + "</td>"));
            tableBody.append(row);
        """),format.raw/*345.9*/("""}"""),format.raw/*345.10*/("""
    """),format.raw/*346.5*/("""}"""),format.raw/*346.6*/("""

    """),format.raw/*348.5*/("""//Compare function for results, based on sort order
    function compareResultsIndex(a, b)"""),format.raw/*349.39*/("""{"""),format.raw/*349.40*/("""
        """),format.raw/*350.9*/("""return (resultTableSortOrder == "ascending" ? a[0] - b[0] : b[0] - a[0]);
    """),format.raw/*351.5*/("""}"""),format.raw/*351.6*/("""
    """),format.raw/*352.5*/("""function compareScores(a,b)"""),format.raw/*352.32*/("""{"""),format.raw/*352.33*/("""
        """),format.raw/*353.9*/("""//TODO Not always numbers...
        if(resultTableSortOrder == "ascending")"""),format.raw/*354.48*/("""{"""),format.raw/*354.49*/("""
            """),format.raw/*355.13*/("""return a[1] - b[1];
        """),format.raw/*356.9*/("""}"""),format.raw/*356.10*/(""" """),format.raw/*356.11*/("""else """),format.raw/*356.16*/("""{"""),format.raw/*356.17*/("""
            """),format.raw/*357.13*/("""return b[1] - a[1];
        """),format.raw/*358.9*/("""}"""),format.raw/*358.10*/("""
    """),format.raw/*359.5*/("""}"""),format.raw/*359.6*/("""
    """),format.raw/*360.5*/("""function compareStatus(a,b)"""),format.raw/*360.32*/("""{"""),format.raw/*360.33*/("""
        """),format.raw/*361.9*/("""//TODO: secondary sort on... score? index?
        if(resultTableSortOrder == "ascending")"""),format.raw/*362.48*/("""{"""),format.raw/*362.49*/("""
            """),format.raw/*363.13*/("""return (a[2] < b[2] ? -1 : (a[2] > b[2] ? 1 : 0));
        """),format.raw/*364.9*/("""}"""),format.raw/*364.10*/(""" """),format.raw/*364.11*/("""else """),format.raw/*364.16*/("""{"""),format.raw/*364.17*/("""
            """),format.raw/*365.13*/("""return (a[2] < b[2] ? 1 : (a[2] > b[2] ? -1 : 0));
        """),format.raw/*366.9*/("""}"""),format.raw/*366.10*/("""
    """),format.raw/*367.5*/("""}"""),format.raw/*367.6*/("""

    """),format.raw/*369.5*/("""//Do a HTTP request on the specified path, parse and insert into the provided element
    function loadCandidateDetails(path, elementToAppendTo)"""),format.raw/*370.59*/("""{"""),format.raw/*370.60*/("""
        """),format.raw/*371.9*/("""$.get(path, function (data) """),format.raw/*371.37*/("""{"""),format.raw/*371.38*/("""
            """),format.raw/*372.13*/("""var str = JSON.stringify(data);
            var component = Component.getComponent(str);
            component.render(elementToAppendTo);
        """),format.raw/*375.9*/("""}"""),format.raw/*375.10*/(""");
    """),format.raw/*376.5*/("""}"""),format.raw/*376.6*/("""



    """),format.raw/*380.5*/("""//Sorting by column: Intercept click events on table header
    $(function()"""),format.raw/*381.17*/("""{"""),format.raw/*381.18*/("""
        """),format.raw/*382.9*/("""$("#resultsTableHeader").delegate("th", "click", function(e) """),format.raw/*382.70*/("""{"""),format.raw/*382.71*/("""
            """),format.raw/*383.13*/("""//console.log("Header clicked on at: " + $(e.currentTarget).index() + " - " + $(e.currentTarget).html());
            //Update the sort order for the table:
            var clickIndex = $(e.currentTarget).index();
            if(clickIndex == resultTableSortIndex)"""),format.raw/*386.51*/("""{"""),format.raw/*386.52*/("""
                """),format.raw/*387.17*/("""//Switch sort order: ascending -> descending or descending -> ascending
                if(resultTableSortOrder == "ascending")"""),format.raw/*388.56*/("""{"""),format.raw/*388.57*/("""
                    """),format.raw/*389.21*/("""resultTableSortOrder = "descending";
                """),format.raw/*390.17*/("""}"""),format.raw/*390.18*/(""" """),format.raw/*390.19*/("""else """),format.raw/*390.24*/("""{"""),format.raw/*390.25*/("""
                    """),format.raw/*391.21*/("""resultTableSortOrder = "ascending";
                """),format.raw/*392.17*/("""}"""),format.raw/*392.18*/("""
            """),format.raw/*393.13*/("""}"""),format.raw/*393.14*/(""" """),format.raw/*393.15*/("""else """),format.raw/*393.20*/("""{"""),format.raw/*393.21*/("""
                """),format.raw/*394.17*/("""//Sort on column, ascending:
                resultTableSortIndex = clickIndex;
                resultTableSortOrder = "ascending";
            """),format.raw/*397.13*/("""}"""),format.raw/*397.14*/("""

            """),format.raw/*399.13*/("""//Clear record of expanded rows
            expandedRowsCandidateIDs = [];

            //Redraw table
            drawResultTable();
        """),format.raw/*404.9*/("""}"""),format.raw/*404.10*/(""");
    """),format.raw/*405.5*/("""}"""),format.raw/*405.6*/(""");

    //Displaying model/candidate details: Intercept click events on table rows -> toggle selected, fire off update
    $(function()"""),format.raw/*408.17*/("""{"""),format.raw/*408.18*/("""
        """),format.raw/*409.9*/("""$("#resultsTableBody").delegate("tr", "click", function(e)"""),format.raw/*409.67*/("""{"""),format.raw/*409.68*/("""
            """),format.raw/*410.13*/("""var id = this.id;   //Expect: rTbl-X  where X is some index
            var dashIdx = id.indexOf("-");
            var candidateID = Number(id.substring(dashIdx+1));

//            console.log("Clicked row: " + this.id + " with class: " + this.className + ", candidateId = " + candidateID);

            if(this.className == "resultTableRow")"""),format.raw/*416.51*/("""{"""),format.raw/*416.52*/("""
                """),format.raw/*417.17*/("""//Set selected model
                selectedCandidateIdx = candidateID;

                //Fire off update
                doUpdate();
            """),format.raw/*422.13*/("""}"""),format.raw/*422.14*/("""
        """),format.raw/*423.9*/("""}"""),format.raw/*423.10*/(""");
    """),format.raw/*424.5*/("""}"""),format.raw/*424.6*/(""");

</script>
<script>
    $(function () """),format.raw/*428.19*/("""{"""),format.raw/*428.20*/("""
        """),format.raw/*429.9*/("""$("#accordion").accordion("""),format.raw/*429.35*/("""{"""),format.raw/*429.36*/("""
            """),format.raw/*430.13*/("""collapsible: true,
            heightStyle: "content"
        """),format.raw/*432.9*/("""}"""),format.raw/*432.10*/(""");
    """),format.raw/*433.5*/("""}"""),format.raw/*433.6*/(""");
    $(function () """),format.raw/*434.19*/("""{"""),format.raw/*434.20*/("""
        """),format.raw/*435.9*/("""$("#accordion2").accordion("""),format.raw/*435.36*/("""{"""),format.raw/*435.37*/("""
            """),format.raw/*436.13*/("""collapsible: true,
            heightStyle: "content"
        """),format.raw/*438.9*/("""}"""),format.raw/*438.10*/(""");
    """),format.raw/*439.5*/("""}"""),format.raw/*439.6*/(""");
    $(function () """),format.raw/*440.19*/("""{"""),format.raw/*440.20*/("""
        """),format.raw/*441.9*/("""$("#accordion3").accordion("""),format.raw/*441.36*/("""{"""),format.raw/*441.37*/("""
            """),format.raw/*442.13*/("""collapsible: true,
            heightStyle: "content"
        """),format.raw/*444.9*/("""}"""),format.raw/*444.10*/(""");
    """),format.raw/*445.5*/("""}"""),format.raw/*445.6*/(""");
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
                  DATE: Fri Jul 21 21:09:59 AEST 2017
                  SOURCE: C:/DL4J/Git/Arbiter/arbiter-ui/src/main/views/org/deeplearning4j/arbiter/ui/views/ArbiterUI.scala.html
                  HASH: 5838e12b1f09b64d92c3473509bc950af56431ea
                  MATRIX: 647->0|1031->356|1060->357|1102->371|1224->466|1253->467|1292->479|1331->490|1360->491|1402->505|1503->579|1532->580|1571->592|1608->601|1637->602|1679->616|1742->652|1771->653|1810->665|1841->668|1870->669|1912->683|2164->908|2193->909|2232->921|2263->924|2292->925|2334->939|2588->1166|2617->1167|2656->1179|2703->1198|2732->1199|2774->1213|2946->1358|2975->1359|3014->1371|3110->1439|3139->1440|3181->1454|3323->1569|3352->1570|3391->1582|3441->1604|3470->1605|3512->1619|3683->1763|3712->1764|3751->1776|3801->1798|3830->1799|3872->1813|4002->1916|4031->1917|4070->1929|4230->2061|4259->2062|4301->2076|4454->2202|4483->2203|4520->2213|4577->2242|4606->2243|4648->2257|4904->2486|4933->2487|4972->2499|5058->2557|5087->2558|5129->2572|5222->2638|5251->2639|5290->2651|5427->2760|5456->2761|5498->2775|5577->2827|5606->2828|5645->2840|5692->2859|5721->2860|5763->2874|6363->3446|6393->3447|6433->3459|6480->3477|6510->3478|6553->3492|6612->3523|6642->3524|6682->3536|6735->3560|6765->3561|6808->3575|6867->3606|6897->3607|6937->3619|7067->3719|7098->3720|7141->3734|7596->4161|7626->4162|7666->4174|7728->4207|7758->4208|7801->4222|8132->4525|8162->4526|8202->4538|8264->4571|8294->4572|8337->4586|8448->4669|8478->4670|8516->4680|8568->4703|8598->4704|8641->4718|8759->4808|8789->4809|8827->4819|8867->4830|8897->4831|8940->4845|9033->4910|9063->4911|9103->4923|9386->5259|9424->5269|10352->6168|10382->6169|10420->6179|10551->6281|10581->6282|10624->6296|11240->6883|11270->6884|11317->6902|11451->7007|11481->7008|11532->7030|11844->7313|11874->7314|11973->7384|12003->7385|12048->7401|12172->7496|12202->7497|12249->7515|12359->7596|12389->7597|12440->7619|12735->7885|12765->7886|12868->7960|12898->7961|12943->7977|13098->8103|13128->8104|13177->8124|13312->8230|13342->8231|13393->8253|13581->8412|13611->8413|13712->8485|13742->8486|13787->8502|13898->8584|13928->8585|13975->8603|14114->8713|14144->8714|14195->8736|14508->9020|14538->9021|14583->9037|14613->9038|14651->9048|14681->9049|14716->9056|14745->9057|14781->9065|14857->9112|14887->9113|14925->9123|15182->9351|15212->9352|15255->9366|15331->9414|15361->9415|15401->9427|15632->9629|15662->9630|15705->9644|15838->9748|15868->9749|15915->9767|16008->9831|16038->9832|16081->9846|16186->9923|16216->9924|16256->9936|16295->9946|16325->9947|16368->9961|16425->9989|16455->9990|16502->10008|16684->10161|16714->10162|16765->10184|16855->10245|16885->10246|16932->10264|17034->10337|17064->10338|17102->10348|17132->10349|17172->10361|17267->10428|17296->10429|17332->10437|17387->10463|17417->10464|17457->10476|18814->11804|18844->11805|18874->11807|19068->11972|19098->11973|19145->11991|19299->12116|19329->12117|19359->12118|19393->12123|19423->12124|19470->12142|19614->12257|19644->12258|19687->12272|20000->12557|20030->12558|20064->12564|20093->12565|20129->12573|20249->12664|20279->12665|20317->12675|20424->12754|20453->12755|20487->12761|20543->12788|20573->12789|20611->12799|20717->12876|20747->12877|20790->12891|20847->12920|20877->12921|20907->12922|20941->12927|20971->12928|21014->12942|21071->12971|21101->12972|21135->12978|21164->12979|21198->12985|21254->13012|21284->13013|21322->13023|21442->13114|21472->13115|21515->13129|21603->13189|21633->13190|21663->13191|21697->13196|21727->13197|21770->13211|21858->13271|21888->13272|21922->13278|21951->13279|21987->13287|22161->13432|22191->13433|22229->13443|22286->13471|22316->13472|22359->13486|22536->13635|22566->13636|22602->13644|22631->13645|22671->13657|22777->13734|22807->13735|22845->13745|22935->13806|22965->13807|23008->13821|23304->14088|23334->14089|23381->14107|23538->14235|23568->14236|23619->14258|23702->14312|23732->14313|23762->14314|23796->14319|23826->14320|23877->14342|23959->14395|23989->14396|24032->14410|24062->14411|24092->14412|24126->14417|24156->14418|24203->14436|24379->14583|24409->14584|24454->14600|24629->14747|24659->14748|24695->14756|24724->14757|24891->14895|24921->14896|24959->14906|25046->14964|25076->14965|25119->14979|25496->15327|25526->15328|25573->15346|25755->15499|25785->15500|25823->15510|25853->15511|25889->15519|25918->15520|25992->15565|26022->15566|26060->15576|26115->15602|26145->15603|26188->15617|26280->15681|26310->15682|26346->15690|26375->15691|26426->15713|26456->15714|26494->15724|26550->15751|26580->15752|26623->15766|26715->15830|26745->15831|26781->15839|26810->15840|26861->15862|26891->15863|26929->15873|26985->15900|27015->15901|27058->15915|27150->15979|27180->15980|27216->15988|27245->15989
                  LINES: 25->1|36->12|36->12|37->13|40->16|40->16|42->18|42->18|42->18|43->19|46->22|46->22|48->24|48->24|48->24|49->25|50->26|50->26|52->28|52->28|52->28|53->29|59->35|59->35|61->37|61->37|61->37|62->38|68->44|68->44|70->46|70->46|70->46|71->47|75->51|75->51|77->53|77->53|77->53|78->54|80->56|80->56|82->58|82->58|82->58|83->59|87->63|87->63|89->65|89->65|89->65|90->66|93->69|93->69|95->71|96->72|96->72|97->73|99->75|99->75|100->76|100->76|100->76|101->77|106->82|106->82|108->84|109->85|109->85|110->86|112->88|112->88|114->90|115->91|115->91|116->92|117->93|117->93|119->95|119->95|119->95|120->96|136->112|136->112|138->114|138->114|138->114|139->115|140->116|140->116|142->118|142->118|142->118|143->119|144->120|144->120|146->122|146->122|146->122|147->123|155->131|155->131|157->133|157->133|157->133|158->134|164->140|164->140|166->142|167->143|167->143|168->144|171->147|171->147|172->148|172->148|172->148|173->149|176->152|176->152|177->153|177->153|177->153|178->154|180->156|180->156|182->158|190->166|191->167|215->191|215->191|216->192|217->193|217->193|218->194|227->203|227->203|228->204|229->205|229->205|230->206|236->212|236->212|239->215|239->215|241->217|242->218|242->218|243->219|244->220|244->220|245->221|252->228|252->228|255->231|255->231|257->233|258->234|258->234|260->236|261->237|261->237|262->238|265->241|265->241|268->244|268->244|270->246|271->247|271->247|272->248|273->249|273->249|274->250|281->257|281->257|282->258|282->258|283->259|283->259|284->260|284->260|286->262|286->262|286->262|287->263|293->269|293->269|294->270|295->271|295->271|297->273|300->276|300->276|301->277|303->279|303->279|304->280|305->281|305->281|306->282|308->284|308->284|310->286|310->286|310->286|311->287|311->287|311->287|312->288|315->291|315->291|316->292|317->293|317->293|318->294|320->296|320->296|321->297|321->297|323->299|325->301|325->301|327->303|327->303|327->303|329->305|354->330|354->330|355->331|357->333|357->333|358->334|360->336|360->336|360->336|360->336|360->336|361->337|363->339|363->339|364->340|369->345|369->345|370->346|370->346|372->348|373->349|373->349|374->350|375->351|375->351|376->352|376->352|376->352|377->353|378->354|378->354|379->355|380->356|380->356|380->356|380->356|380->356|381->357|382->358|382->358|383->359|383->359|384->360|384->360|384->360|385->361|386->362|386->362|387->363|388->364|388->364|388->364|388->364|388->364|389->365|390->366|390->366|391->367|391->367|393->369|394->370|394->370|395->371|395->371|395->371|396->372|399->375|399->375|400->376|400->376|404->380|405->381|405->381|406->382|406->382|406->382|407->383|410->386|410->386|411->387|412->388|412->388|413->389|414->390|414->390|414->390|414->390|414->390|415->391|416->392|416->392|417->393|417->393|417->393|417->393|417->393|418->394|421->397|421->397|423->399|428->404|428->404|429->405|429->405|432->408|432->408|433->409|433->409|433->409|434->410|440->416|440->416|441->417|446->422|446->422|447->423|447->423|448->424|448->424|452->428|452->428|453->429|453->429|453->429|454->430|456->432|456->432|457->433|457->433|458->434|458->434|459->435|459->435|459->435|460->436|462->438|462->438|463->439|463->439|464->440|464->440|465->441|465->441|465->441|466->442|468->444|468->444|469->445|469->445
                  -- GENERATED --
              */
          