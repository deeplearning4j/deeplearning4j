
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
        <link rel="stylesheet" href="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap.min.css">

        <script src="//ajax.googleapis.com/ajax/libs/jquery/1/jquery.min.js"></script>
        <link rel="stylesheet" href="//code.jquery.com/ui/1.11.4/themes/smoothness/jquery-ui.css">
        <script src="//code.jquery.com/jquery-1.10.2.js"></script>
        <script src="//code.jquery.com/ui/1.11.4/jquery-ui.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js"></script>
        <script src="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/js/bootstrap.min.js"></script>
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
    setInterval(function()"""),format.raw/*187.27*/("""{"""),format.raw/*187.28*/("""
        """),format.raw/*188.9*/("""//Get the update status, and do something with it:
        $.get("/arbiter/lastUpdate",function(data)"""),format.raw/*189.51*/("""{"""),format.raw/*189.52*/("""
            """),format.raw/*190.13*/("""//Encoding: matches names in UpdateStatus class
            var jsonObj = JSON.parse(JSON.stringify(data));
            var statusTime = jsonObj['statusUpdateTime'];
            var settingsTime = jsonObj['settingsUpdateTime'];
            var resultsTime = jsonObj['resultsUpdateTime'];
            //console.log("Last update times: " + statusTime + ", " + settingsTime + ", " + resultsTime);

            //Check last update times for each part of document, and update as necessary
            //First section: summary status
            if(lastStatusUpdateTime != statusTime)"""),format.raw/*199.51*/("""{"""),format.raw/*199.52*/("""
                """),format.raw/*200.17*/("""//Get JSON: address set by SummaryStatusResource
                $.get("/arbiter/summary",function(data)"""),format.raw/*201.56*/("""{"""),format.raw/*201.57*/("""
                    """),format.raw/*202.21*/("""var summaryStatusDiv = $('#statusdiv');
                    summaryStatusDiv.html('');

                    var str = JSON.stringify(data);
                    var component = Component.getComponent(str);
                    component.render(summaryStatusDiv);
                """),format.raw/*208.17*/("""}"""),format.raw/*208.18*/(""");

                lastStatusUpdateTime = statusTime;
            """),format.raw/*211.13*/("""}"""),format.raw/*211.14*/("""

            """),format.raw/*213.13*/("""//Second section: Optimization settings
            if(lastSettingsUpdateTime != settingsTime)"""),format.raw/*214.55*/("""{"""),format.raw/*214.56*/("""
                """),format.raw/*215.17*/("""//Get JSON for components
                $.get("/arbiter/config",function(data)"""),format.raw/*216.55*/("""{"""),format.raw/*216.56*/("""
                    """),format.raw/*217.21*/("""var str = JSON.stringify(data);

                    var configDiv = $('#settingsdiv');
                    configDiv.html('');

                    var component = Component.getComponent(str);
                    component.render(configDiv);
                """),format.raw/*224.17*/("""}"""),format.raw/*224.18*/(""");

                lastSettingsUpdateTime = settingsTime;
            """),format.raw/*227.13*/("""}"""),format.raw/*227.14*/("""

            """),format.raw/*229.13*/("""//Third section: Summary results table (summary info for each candidate)
            if(lastResultsUpdateTime != resultsTime)"""),format.raw/*230.53*/("""{"""),format.raw/*230.54*/("""

                """),format.raw/*232.17*/("""//Get JSON; address set by SummaryResultsResource
                $.get("/arbiter/results",function(data)"""),format.raw/*233.56*/("""{"""),format.raw/*233.57*/("""
                    """),format.raw/*234.21*/("""//Expect an array of CandidateInfo type objects here
                    resultsTableContent = data;
                    drawResultTable();
                """),format.raw/*237.17*/("""}"""),format.raw/*237.18*/(""");

                lastResultsUpdateTime = resultsTime;
            """),format.raw/*240.13*/("""}"""),format.raw/*240.14*/("""

            """),format.raw/*242.13*/("""//Finally: Currently selected result
            if(selectedCandidateIdx != null)"""),format.raw/*243.45*/("""{"""),format.raw/*243.46*/("""
                """),format.raw/*244.17*/("""//Get JSON for components
                $.get("/arbiter/candidateInfo/"+selectedCandidateIdx,function(data)"""),format.raw/*245.84*/("""{"""),format.raw/*245.85*/("""
                    """),format.raw/*246.21*/("""var str = JSON.stringify(data);

                    var resultsViewDiv = $('#resultsviewdiv');
                    resultsViewDiv.html('');

                    var component = Component.getComponent(str);
                    component.render(resultsViewDiv);
                """),format.raw/*253.17*/("""}"""),format.raw/*253.18*/(""");
            """),format.raw/*254.13*/("""}"""),format.raw/*254.14*/("""
        """),format.raw/*255.9*/("""}"""),format.raw/*255.10*/(""")
    """),format.raw/*256.5*/("""}"""),format.raw/*256.6*/(""",2000);    //Loop every 2 seconds

    function createTable(tableObj,tableId,appendTo)"""),format.raw/*258.52*/("""{"""),format.raw/*258.53*/("""
        """),format.raw/*259.9*/("""//Expect RenderableComponentTable
        var header = tableObj['header'];
        var values = tableObj['table'];
        var title = tableObj['title'];
        var nRows = (values ? values.length : 0);

        if(title)"""),format.raw/*265.18*/("""{"""),format.raw/*265.19*/("""
            """),format.raw/*266.13*/("""appendTo.append("<h5>"+title+"</h5>");
        """),format.raw/*267.9*/("""}"""),format.raw/*267.10*/("""

        """),format.raw/*269.9*/("""var table;
        if(tableId) table = $("<table id=\"" + tableId + "\" class=\"renderableComponentTable\">");
        else table = $("<table class=\"renderableComponentTable\">");
        if(header)"""),format.raw/*272.19*/("""{"""),format.raw/*272.20*/("""
            """),format.raw/*273.13*/("""var headerRow = $("<tr>");
            var len = header.length;
            for( var i=0; i<len; i++ )"""),format.raw/*275.39*/("""{"""),format.raw/*275.40*/("""
                """),format.raw/*276.17*/("""headerRow.append($("<th>" + header[i] + "</th>"));
            """),format.raw/*277.13*/("""}"""),format.raw/*277.14*/("""
            """),format.raw/*278.13*/("""headerRow.append($("</tr>"));
            table.append(headerRow);
        """),format.raw/*280.9*/("""}"""),format.raw/*280.10*/("""

        """),format.raw/*282.9*/("""if(values)"""),format.raw/*282.19*/("""{"""),format.raw/*282.20*/("""
            """),format.raw/*283.13*/("""for( var i=0; i<nRows; i++ )"""),format.raw/*283.41*/("""{"""),format.raw/*283.42*/("""
                """),format.raw/*284.17*/("""var row = $("<tr>");
                var rowValues = values[i];
                var len = rowValues.length;
                for( var j=0; j<len; j++ )"""),format.raw/*287.43*/("""{"""),format.raw/*287.44*/("""
                    """),format.raw/*288.21*/("""row.append($('<td>'+rowValues[j]+'</td>'));
                """),format.raw/*289.17*/("""}"""),format.raw/*289.18*/("""
                """),format.raw/*290.17*/("""row.append($("</tr>"));
                table.append(row);
            """),format.raw/*292.13*/("""}"""),format.raw/*292.14*/("""
        """),format.raw/*293.9*/("""}"""),format.raw/*293.10*/("""

        """),format.raw/*295.9*/("""table.append($("</table>"));
        appendTo.append(table);
    """),format.raw/*297.5*/("""}"""),format.raw/*297.6*/("""

    """),format.raw/*299.5*/("""function drawResultTable()"""),format.raw/*299.31*/("""{"""),format.raw/*299.32*/("""

        """),format.raw/*301.9*/("""//Remove all elements from the table body
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
        for(var i=0; i<len; i++)"""),format.raw/*326.33*/("""{"""),format.raw/*326.34*/("""
"""),format.raw/*327.1*/("""//            var row = $('<tr class="resultTableRow" id="rTbl-' + sorted[i][0] + '"/>');
            var row;
            if(selectedCandidateIdx == sorted[i][0])"""),format.raw/*329.53*/("""{"""),format.raw/*329.54*/("""
                """),format.raw/*330.17*/("""//Selected row
                row = $('<tr class="resultTableRowSelected" id="rTbl-' + sorted[i][0] + '"/>');
            """),format.raw/*332.13*/("""}"""),format.raw/*332.14*/(""" """),format.raw/*332.15*/("""else """),format.raw/*332.20*/("""{"""),format.raw/*332.21*/("""
                """),format.raw/*333.17*/("""//Normal row
                row = $('<tr class="resultTableRow" id="rTbl-' + sorted[i][0] + '"/>');
            """),format.raw/*335.13*/("""}"""),format.raw/*335.14*/("""
            """),format.raw/*336.13*/("""row.append($("<td>" + sorted[i][0] + "</td>"));
            var score = sorted[i][1];
            row.append($("<td>" + ((!score || score == "null") ? "-" : score) + "</td>"));
            row.append($("<td>" + sorted[i][2] + "</td>"));
            tableBody.append(row);
        """),format.raw/*341.9*/("""}"""),format.raw/*341.10*/("""
    """),format.raw/*342.5*/("""}"""),format.raw/*342.6*/("""

    """),format.raw/*344.5*/("""//Compare function for results, based on sort order
    function compareResultsIndex(a, b)"""),format.raw/*345.39*/("""{"""),format.raw/*345.40*/("""
        """),format.raw/*346.9*/("""return (resultTableSortOrder == "ascending" ? a[0] - b[0] : b[0] - a[0]);
    """),format.raw/*347.5*/("""}"""),format.raw/*347.6*/("""
    """),format.raw/*348.5*/("""function compareScores(a,b)"""),format.raw/*348.32*/("""{"""),format.raw/*348.33*/("""
        """),format.raw/*349.9*/("""//TODO Not always numbers...
        if(resultTableSortOrder == "ascending")"""),format.raw/*350.48*/("""{"""),format.raw/*350.49*/("""
            """),format.raw/*351.13*/("""return a[1] - b[1];
        """),format.raw/*352.9*/("""}"""),format.raw/*352.10*/(""" """),format.raw/*352.11*/("""else """),format.raw/*352.16*/("""{"""),format.raw/*352.17*/("""
            """),format.raw/*353.13*/("""return b[1] - a[1];
        """),format.raw/*354.9*/("""}"""),format.raw/*354.10*/("""
    """),format.raw/*355.5*/("""}"""),format.raw/*355.6*/("""
    """),format.raw/*356.5*/("""function compareStatus(a,b)"""),format.raw/*356.32*/("""{"""),format.raw/*356.33*/("""
        """),format.raw/*357.9*/("""//TODO: secondary sort on... score? index?
        if(resultTableSortOrder == "ascending")"""),format.raw/*358.48*/("""{"""),format.raw/*358.49*/("""
            """),format.raw/*359.13*/("""return (a[2] < b[2] ? -1 : (a[2] > b[2] ? 1 : 0));
        """),format.raw/*360.9*/("""}"""),format.raw/*360.10*/(""" """),format.raw/*360.11*/("""else """),format.raw/*360.16*/("""{"""),format.raw/*360.17*/("""
            """),format.raw/*361.13*/("""return (a[2] < b[2] ? 1 : (a[2] > b[2] ? -1 : 0));
        """),format.raw/*362.9*/("""}"""),format.raw/*362.10*/("""
    """),format.raw/*363.5*/("""}"""),format.raw/*363.6*/("""

    """),format.raw/*365.5*/("""//Do a HTTP request on the specified path, parse and insert into the provided element
    function loadCandidateDetails(path, elementToAppendTo)"""),format.raw/*366.59*/("""{"""),format.raw/*366.60*/("""
        """),format.raw/*367.9*/("""$.get(path, function (data) """),format.raw/*367.37*/("""{"""),format.raw/*367.38*/("""
            """),format.raw/*368.13*/("""var str = JSON.stringify(data);
            var component = Component.getComponent(str);
            component.render(elementToAppendTo);
        """),format.raw/*371.9*/("""}"""),format.raw/*371.10*/(""");
    """),format.raw/*372.5*/("""}"""),format.raw/*372.6*/("""



    """),format.raw/*376.5*/("""//Sorting by column: Intercept click events on table header
    $(function()"""),format.raw/*377.17*/("""{"""),format.raw/*377.18*/("""
        """),format.raw/*378.9*/("""$("#resultsTableHeader").delegate("th", "click", function(e) """),format.raw/*378.70*/("""{"""),format.raw/*378.71*/("""
            """),format.raw/*379.13*/("""//console.log("Header clicked on at: " + $(e.currentTarget).index() + " - " + $(e.currentTarget).html());
            //Update the sort order for the table:
            var clickIndex = $(e.currentTarget).index();
            if(clickIndex == resultTableSortIndex)"""),format.raw/*382.51*/("""{"""),format.raw/*382.52*/("""
                """),format.raw/*383.17*/("""//Switch sort order: ascending -> descending or descending -> ascending
                if(resultTableSortOrder == "ascending")"""),format.raw/*384.56*/("""{"""),format.raw/*384.57*/("""
                    """),format.raw/*385.21*/("""resultTableSortOrder = "descending";
                """),format.raw/*386.17*/("""}"""),format.raw/*386.18*/(""" """),format.raw/*386.19*/("""else """),format.raw/*386.24*/("""{"""),format.raw/*386.25*/("""
                    """),format.raw/*387.21*/("""resultTableSortOrder = "ascending";
                """),format.raw/*388.17*/("""}"""),format.raw/*388.18*/("""
            """),format.raw/*389.13*/("""}"""),format.raw/*389.14*/(""" """),format.raw/*389.15*/("""else """),format.raw/*389.20*/("""{"""),format.raw/*389.21*/("""
                """),format.raw/*390.17*/("""//Sort on column, ascending:
                resultTableSortIndex = clickIndex;
                resultTableSortOrder = "ascending";
            """),format.raw/*393.13*/("""}"""),format.raw/*393.14*/("""

            """),format.raw/*395.13*/("""//Clear record of expanded rows
            expandedRowsCandidateIDs = [];

            //Redraw table
            drawResultTable();
        """),format.raw/*400.9*/("""}"""),format.raw/*400.10*/(""");
    """),format.raw/*401.5*/("""}"""),format.raw/*401.6*/(""");

    //Displaying model/candidate details: Intercept click events on table rows -> toggle visibility on content rows
    $(function()"""),format.raw/*404.17*/("""{"""),format.raw/*404.18*/("""
        """),format.raw/*405.9*/("""$("#resultsTableBody").delegate("tr", "click", function(e)"""),format.raw/*405.67*/("""{"""),format.raw/*405.68*/("""
            """),format.raw/*406.13*/("""var id = this.id;   //Expect: rTbl-X  where X is some index
            var dashIdx = id.indexOf("-");
            var candidateID = Number(id.substring(dashIdx+1));

            console.log("Clicked row: " + this.id + " with class: " + this.className + ", candidateId = " + candidateID);

            if(this.className == "resultTableRow")"""),format.raw/*412.51*/("""{"""),format.raw/*412.52*/("""
                """),format.raw/*413.17*/("""//Set selected model
                selectedCandidateIdx = candidateID;

                //TODO fire off update
            """),format.raw/*417.13*/("""}"""),format.raw/*417.14*/("""
        """),format.raw/*418.9*/("""}"""),format.raw/*418.10*/(""");
    """),format.raw/*419.5*/("""}"""),format.raw/*419.6*/(""");

</script>
<script>
    $(function () """),format.raw/*423.19*/("""{"""),format.raw/*423.20*/("""
        """),format.raw/*424.9*/("""$("#accordion").accordion("""),format.raw/*424.35*/("""{"""),format.raw/*424.36*/("""
            """),format.raw/*425.13*/("""collapsible: true,
            heightStyle: "content"
        """),format.raw/*427.9*/("""}"""),format.raw/*427.10*/(""");
    """),format.raw/*428.5*/("""}"""),format.raw/*428.6*/(""");
    $(function () """),format.raw/*429.19*/("""{"""),format.raw/*429.20*/("""
        """),format.raw/*430.9*/("""$("#accordion2").accordion("""),format.raw/*430.36*/("""{"""),format.raw/*430.37*/("""
            """),format.raw/*431.13*/("""collapsible: true,
            heightStyle: "content"
        """),format.raw/*433.9*/("""}"""),format.raw/*433.10*/(""");
    """),format.raw/*434.5*/("""}"""),format.raw/*434.6*/(""");
    $(function () """),format.raw/*435.19*/("""{"""),format.raw/*435.20*/("""
        """),format.raw/*436.9*/("""$("#accordion3").accordion("""),format.raw/*436.36*/("""{"""),format.raw/*436.37*/("""
            """),format.raw/*437.13*/("""collapsible: true,
            heightStyle: "content"
        """),format.raw/*439.9*/("""}"""),format.raw/*439.10*/(""");
    """),format.raw/*440.5*/("""}"""),format.raw/*440.6*/(""");
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
                  DATE: Fri Jul 21 13:28:10 AEST 2017
                  SOURCE: C:/DL4J/Git/Arbiter/arbiter-ui/src/main/views/org/deeplearning4j/arbiter/ui/views/ArbiterUI.scala.html
                  HASH: f0d8d3ffdaa63de81b259cd23326f41365a4aeb9
                  MATRIX: 647->0|1031->356|1060->357|1102->371|1224->466|1253->467|1292->479|1331->490|1360->491|1402->505|1503->579|1532->580|1571->592|1608->601|1637->602|1679->616|1742->652|1771->653|1810->665|1841->668|1870->669|1912->683|2164->908|2193->909|2232->921|2263->924|2292->925|2334->939|2588->1166|2617->1167|2656->1179|2703->1198|2732->1199|2774->1213|2946->1358|2975->1359|3014->1371|3110->1439|3139->1440|3181->1454|3323->1569|3352->1570|3391->1582|3441->1604|3470->1605|3512->1619|3683->1763|3712->1764|3751->1776|3801->1798|3830->1799|3872->1813|4002->1916|4031->1917|4070->1929|4230->2061|4259->2062|4301->2076|4454->2202|4483->2203|4520->2213|4577->2242|4606->2243|4648->2257|4904->2486|4933->2487|4972->2499|5058->2557|5087->2558|5129->2572|5222->2638|5251->2639|5290->2651|5427->2760|5456->2761|5498->2775|5577->2827|5606->2828|5645->2840|5692->2859|5721->2860|5763->2874|6363->3446|6393->3447|6433->3459|6480->3477|6510->3478|6553->3492|6612->3523|6642->3524|6682->3536|6735->3560|6765->3561|6808->3575|6867->3606|6897->3607|6937->3619|7067->3719|7098->3720|7141->3734|7596->4161|7626->4162|7666->4174|7728->4207|7758->4208|7801->4222|8132->4525|8162->4526|8202->4538|8264->4571|8294->4572|8337->4586|8448->4669|8478->4670|8516->4680|8568->4703|8598->4704|8641->4718|8759->4808|8789->4809|8827->4819|8867->4830|8897->4831|8940->4845|9033->4910|9063->4911|9103->4923|10374->6165|10404->6166|10442->6176|10573->6278|10603->6279|10646->6293|11262->6880|11292->6881|11339->6899|11473->7004|11503->7005|11554->7027|11866->7310|11896->7311|11995->7381|12025->7382|12070->7398|12194->7493|12224->7494|12271->7512|12381->7593|12411->7594|12462->7616|12757->7882|12787->7883|12890->7957|12920->7958|12965->7974|13120->8100|13150->8101|13199->8121|13334->8227|13364->8228|13415->8250|13603->8409|13633->8410|13734->8482|13764->8483|13809->8499|13920->8581|13950->8582|13997->8600|14136->8710|14166->8711|14217->8733|14530->9017|14560->9018|14605->9034|14635->9035|14673->9045|14703->9046|14738->9053|14767->9054|14884->9142|14914->9143|14952->9153|15209->9381|15239->9382|15282->9396|15358->9444|15388->9445|15428->9457|15659->9659|15689->9660|15732->9674|15865->9778|15895->9779|15942->9797|16035->9861|16065->9862|16108->9876|16213->9953|16243->9954|16283->9966|16322->9976|16352->9977|16395->9991|16452->10019|16482->10020|16529->10038|16711->10191|16741->10192|16792->10214|16882->10275|16912->10276|16959->10294|17061->10367|17091->10368|17129->10378|17159->10379|17199->10391|17294->10458|17323->10459|17359->10467|17414->10493|17444->10494|17484->10506|18841->11834|18871->11835|18901->11837|19095->12002|19125->12003|19172->12021|19326->12146|19356->12147|19386->12148|19420->12153|19450->12154|19497->12172|19641->12287|19671->12288|19714->12302|20027->12587|20057->12588|20091->12594|20120->12595|20156->12603|20276->12694|20306->12695|20344->12705|20451->12784|20480->12785|20514->12791|20570->12818|20600->12819|20638->12829|20744->12906|20774->12907|20817->12921|20874->12950|20904->12951|20934->12952|20968->12957|20998->12958|21041->12972|21098->13001|21128->13002|21162->13008|21191->13009|21225->13015|21281->13042|21311->13043|21349->13053|21469->13144|21499->13145|21542->13159|21630->13219|21660->13220|21690->13221|21724->13226|21754->13227|21797->13241|21885->13301|21915->13302|21949->13308|21978->13309|22014->13317|22188->13462|22218->13463|22256->13473|22313->13501|22343->13502|22386->13516|22563->13665|22593->13666|22629->13674|22658->13675|22698->13687|22804->13764|22834->13765|22872->13775|22962->13836|22992->13837|23035->13851|23331->14118|23361->14119|23408->14137|23565->14265|23595->14266|23646->14288|23729->14342|23759->14343|23789->14344|23823->14349|23853->14350|23904->14372|23986->14425|24016->14426|24059->14440|24089->14441|24119->14442|24153->14447|24183->14448|24230->14466|24406->14613|24436->14614|24481->14630|24656->14777|24686->14778|24722->14786|24751->14787|24919->14926|24949->14927|24987->14937|25074->14995|25104->14996|25147->15010|25522->15356|25552->15357|25599->15375|25757->15504|25787->15505|25825->15515|25855->15516|25891->15524|25920->15525|25994->15570|26024->15571|26062->15581|26117->15607|26147->15608|26190->15622|26282->15686|26312->15687|26348->15695|26377->15696|26428->15718|26458->15719|26496->15729|26552->15756|26582->15757|26625->15771|26717->15835|26747->15836|26783->15844|26812->15845|26863->15867|26893->15868|26931->15878|26987->15905|27017->15906|27060->15920|27152->15984|27182->15985|27218->15993|27247->15994
                  LINES: 25->1|36->12|36->12|37->13|40->16|40->16|42->18|42->18|42->18|43->19|46->22|46->22|48->24|48->24|48->24|49->25|50->26|50->26|52->28|52->28|52->28|53->29|59->35|59->35|61->37|61->37|61->37|62->38|68->44|68->44|70->46|70->46|70->46|71->47|75->51|75->51|77->53|77->53|77->53|78->54|80->56|80->56|82->58|82->58|82->58|83->59|87->63|87->63|89->65|89->65|89->65|90->66|93->69|93->69|95->71|96->72|96->72|97->73|99->75|99->75|100->76|100->76|100->76|101->77|106->82|106->82|108->84|109->85|109->85|110->86|112->88|112->88|114->90|115->91|115->91|116->92|117->93|117->93|119->95|119->95|119->95|120->96|136->112|136->112|138->114|138->114|138->114|139->115|140->116|140->116|142->118|142->118|142->118|143->119|144->120|144->120|146->122|146->122|146->122|147->123|155->131|155->131|157->133|157->133|157->133|158->134|164->140|164->140|166->142|167->143|167->143|168->144|171->147|171->147|172->148|172->148|172->148|173->149|176->152|176->152|177->153|177->153|177->153|178->154|180->156|180->156|182->158|211->187|211->187|212->188|213->189|213->189|214->190|223->199|223->199|224->200|225->201|225->201|226->202|232->208|232->208|235->211|235->211|237->213|238->214|238->214|239->215|240->216|240->216|241->217|248->224|248->224|251->227|251->227|253->229|254->230|254->230|256->232|257->233|257->233|258->234|261->237|261->237|264->240|264->240|266->242|267->243|267->243|268->244|269->245|269->245|270->246|277->253|277->253|278->254|278->254|279->255|279->255|280->256|280->256|282->258|282->258|283->259|289->265|289->265|290->266|291->267|291->267|293->269|296->272|296->272|297->273|299->275|299->275|300->276|301->277|301->277|302->278|304->280|304->280|306->282|306->282|306->282|307->283|307->283|307->283|308->284|311->287|311->287|312->288|313->289|313->289|314->290|316->292|316->292|317->293|317->293|319->295|321->297|321->297|323->299|323->299|323->299|325->301|350->326|350->326|351->327|353->329|353->329|354->330|356->332|356->332|356->332|356->332|356->332|357->333|359->335|359->335|360->336|365->341|365->341|366->342|366->342|368->344|369->345|369->345|370->346|371->347|371->347|372->348|372->348|372->348|373->349|374->350|374->350|375->351|376->352|376->352|376->352|376->352|376->352|377->353|378->354|378->354|379->355|379->355|380->356|380->356|380->356|381->357|382->358|382->358|383->359|384->360|384->360|384->360|384->360|384->360|385->361|386->362|386->362|387->363|387->363|389->365|390->366|390->366|391->367|391->367|391->367|392->368|395->371|395->371|396->372|396->372|400->376|401->377|401->377|402->378|402->378|402->378|403->379|406->382|406->382|407->383|408->384|408->384|409->385|410->386|410->386|410->386|410->386|410->386|411->387|412->388|412->388|413->389|413->389|413->389|413->389|413->389|414->390|417->393|417->393|419->395|424->400|424->400|425->401|425->401|428->404|428->404|429->405|429->405|429->405|430->406|436->412|436->412|437->413|441->417|441->417|442->418|442->418|443->419|443->419|447->423|447->423|448->424|448->424|448->424|449->425|451->427|451->427|452->428|452->428|453->429|453->429|454->430|454->430|454->430|455->431|457->433|457->433|458->434|458->434|459->435|459->435|460->436|460->436|460->436|461->437|463->439|463->439|464->440|464->440
                  -- GENERATED --
              */
          