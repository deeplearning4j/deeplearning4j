
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

        .hd """),format.raw/*11.13*/("""{"""),format.raw/*11.14*/("""
            """),format.raw/*12.13*/("""background-color: #000000;
            height: 41px;
            font-size: 20px;
            color: #FFFFFF;
            font-family: "Open Sans", sans-serif;
            font-weight: 200;
        """),format.raw/*18.9*/("""}"""),format.raw/*18.10*/("""

        """),format.raw/*20.9*/("""html, body """),format.raw/*20.20*/("""{"""),format.raw/*20.21*/("""
            """),format.raw/*21.13*/("""width: 100%;
            height: 100%;
            padding: 0;
        """),format.raw/*24.9*/("""}"""),format.raw/*24.10*/("""

        """),format.raw/*26.9*/(""".bgcolor """),format.raw/*26.18*/("""{"""),format.raw/*26.19*/("""
            """),format.raw/*27.13*/("""background-color: #EFEFEF;
        """),format.raw/*28.9*/("""}"""),format.raw/*28.10*/("""

        """),format.raw/*30.9*/("""h1 """),format.raw/*30.12*/("""{"""),format.raw/*30.13*/("""
            """),format.raw/*31.13*/("""font-family: "Open Sans", sans-serif;
            font-size: 28px;
            font-style: bold;
            font-variant: normal;
            font-weight: 500;
            line-height: 26.4px;
        """),format.raw/*37.9*/("""}"""),format.raw/*37.10*/("""

        """),format.raw/*39.9*/("""h3 """),format.raw/*39.12*/("""{"""),format.raw/*39.13*/("""
            """),format.raw/*40.13*/("""font-family: "Open Sans", sans-serif;
            font-size: 16px;
            font-style: normal;
            font-variant: normal;
            font-weight: 500;
            line-height: 26.4px;
        """),format.raw/*46.9*/("""}"""),format.raw/*46.10*/("""

        """),format.raw/*48.9*/("""table """),format.raw/*48.15*/("""{"""),format.raw/*48.16*/("""
            """),format.raw/*49.13*/("""font-family: "Open Sans", sans-serif;
            font-size: 14px;
        """),format.raw/*51.9*/("""}"""),format.raw/*51.10*/("""

        """),format.raw/*53.9*/("""table.resultsTable """),format.raw/*53.28*/("""{"""),format.raw/*53.29*/("""
            """),format.raw/*54.13*/("""border-collapse:collapse;
            background-color: white;
            /*border-collapse: collapse;*/
            padding: 15px;
        """),format.raw/*58.9*/("""}"""),format.raw/*58.10*/("""

        """),format.raw/*60.9*/("""table.resultsTable td, table.resultsTable tr, table.resultsTable th """),format.raw/*60.77*/("""{"""),format.raw/*60.78*/("""
            """),format.raw/*61.13*/("""border:solid black 1px;
            white-space: pre;   /* assume text is preprocessed for formatting */
        """),format.raw/*63.9*/("""}"""),format.raw/*63.10*/("""

        """),format.raw/*65.9*/("""table.resultsTable th """),format.raw/*65.31*/("""{"""),format.raw/*65.32*/("""
            """),format.raw/*66.13*/("""background-color: /*headingbgcol*/#063E53;
            color: white;
            padding-left: 4px;
            padding-right: 4px;
        """),format.raw/*70.9*/("""}"""),format.raw/*70.10*/("""

        """),format.raw/*72.9*/("""table.resultsTable td """),format.raw/*72.31*/("""{"""),format.raw/*72.32*/("""
            """),format.raw/*73.13*/("""/*background-color: white;*/
            padding-left: 4px;
            padding-right: 4px;
        """),format.raw/*76.9*/("""}"""),format.raw/*76.10*/("""

        """),format.raw/*78.9*/("""/* Properties for table cells in the tables generated using the RenderableComponent mechanism */
        .renderableComponentTable """),format.raw/*79.35*/("""{"""),format.raw/*79.36*/("""
            """),format.raw/*80.13*/("""/*table-layout:fixed; */    /*Avoids scrollbar, but makes fixed width for all columns :( */
            width: 100%
        """),format.raw/*82.9*/("""}"""),format.raw/*82.10*/("""
        """),format.raw/*83.9*/(""".renderableComponentTable td """),format.raw/*83.38*/("""{"""),format.raw/*83.39*/("""
            """),format.raw/*84.13*/("""padding-left: 4px;
            padding-right: 4px;
            white-space: pre;   /* assume text is pre-processed (important for line breaks etc)*/
            word-wrap:break-word;
            vertical-align: top;
        """),format.raw/*89.9*/("""}"""),format.raw/*89.10*/("""

        """),format.raw/*91.9*/("""/** CSS for result table rows */
        .resultTableRow """),format.raw/*92.25*/("""{"""),format.raw/*92.26*/("""
            """),format.raw/*93.13*/("""background-color: #FFFFFF;
            cursor: pointer;
        """),format.raw/*95.9*/("""}"""),format.raw/*95.10*/("""

        """),format.raw/*97.9*/("""/** CSS for result table CONTENT rows (i.e., only visible when expanded) */
        .resultTableRowSelected """),format.raw/*98.33*/("""{"""),format.raw/*98.34*/("""
            """),format.raw/*99.13*/("""background-color: rgba(0, 157, 255, 0.16);
        """),format.raw/*100.9*/("""}"""),format.raw/*100.10*/("""

        """),format.raw/*102.9*/(""".resultsHeadingDiv """),format.raw/*102.28*/("""{"""),format.raw/*102.29*/("""
            """),format.raw/*103.13*/("""background-color: /*headingbgcol*/#063E53;
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
        """),format.raw/*119.9*/("""}"""),format.raw/*119.10*/("""

        """),format.raw/*121.9*/("""div.outerelements """),format.raw/*121.27*/("""{"""),format.raw/*121.28*/("""
            """),format.raw/*122.13*/("""padding-bottom: 30px;
        """),format.raw/*123.9*/("""}"""),format.raw/*123.10*/("""

        """),format.raw/*125.9*/("""#accordion, #accordion2 """),format.raw/*125.33*/("""{"""),format.raw/*125.34*/("""
            """),format.raw/*126.13*/("""padding-bottom: 20px;
        """),format.raw/*127.9*/("""}"""),format.raw/*127.10*/("""

        """),format.raw/*129.9*/("""#accordion .ui-accordion-header, #accordion2 .ui-accordion-header, #accordion3 .ui-accordion-header """),format.raw/*129.109*/("""{"""),format.raw/*129.110*/("""
            """),format.raw/*130.13*/("""background-color: /*headingbgcolor*/#063E53;      /*Color when collapsed*/
            color: /*headingtextcolor*/white;
            font-family: "Open Sans", sans-serif;
            font-size: 16px;
            font-style: bold;
            font-variant: normal;
            margin: 0px;
            background-image: none;     /* Necessary, otherwise color changes don't make a difference */
        """),format.raw/*138.9*/("""}"""),format.raw/*138.10*/("""

        """),format.raw/*140.9*/("""#accordion .ui-accordion-content """),format.raw/*140.42*/("""{"""),format.raw/*140.43*/("""
            """),format.raw/*141.13*/("""width: 100%;
            background-color: white;    /*background color of accordian content (elements in front may have different color */
            color: black;  /* text etc color */
            font-size: 10pt;
            line-height: 16pt;
            overflow:visible !important;
        """),format.raw/*147.9*/("""}"""),format.raw/*147.10*/("""

        """),format.raw/*149.9*/("""/** Line charts */
        path """),format.raw/*150.14*/("""{"""),format.raw/*150.15*/("""
            """),format.raw/*151.13*/("""stroke: steelblue;
            stroke-width: 2;
            fill: none;
        """),format.raw/*154.9*/("""}"""),format.raw/*154.10*/("""
        """),format.raw/*155.9*/(""".axis path, .axis line """),format.raw/*155.32*/("""{"""),format.raw/*155.33*/("""
            """),format.raw/*156.13*/("""fill: none;
            stroke: #000;
            shape-rendering: crispEdges;
        """),format.raw/*159.9*/("""}"""),format.raw/*159.10*/("""
        """),format.raw/*160.9*/(""".tick line """),format.raw/*160.20*/("""{"""),format.raw/*160.21*/("""
            """),format.raw/*161.13*/("""opacity: 0.2;
            shape-rendering: crispEdges;
        """),format.raw/*163.9*/("""}"""),format.raw/*163.10*/("""

        """),format.raw/*165.9*/("""</style>
        <title>DL4J - Arbiter UI</title>
    </head>
    <body class="bgcolor">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="stylesheet" href="/assets/css/arbiter/bootstrap.min.css">
        <script src="/assets/js/jquery-1.9.1.min.js"></script>
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


    function doUpdate()"""),format.raw/*195.24*/("""{"""),format.raw/*195.25*/("""
        """),format.raw/*196.9*/("""//Get the update status, and do something with it:
        $.get("/arbiter/lastUpdate",function(data)"""),format.raw/*197.51*/("""{"""),format.raw/*197.52*/("""
            """),format.raw/*198.13*/("""//Encoding: matches names in UpdateStatus class
            var jsonObj = JSON.parse(JSON.stringify(data));
            var statusTime = jsonObj['statusUpdateTime'];
            var settingsTime = jsonObj['settingsUpdateTime'];
            var resultsTime = jsonObj['resultsUpdateTime'];
            //console.log("Last update times: " + statusTime + ", " + settingsTime + ", " + resultsTime);

            //Check last update times for each part of document, and update as necessary
            //First section: summary status
            if(lastStatusUpdateTime != statusTime)"""),format.raw/*207.51*/("""{"""),format.raw/*207.52*/("""
                """),format.raw/*208.17*/("""//Get JSON: address set by SummaryStatusResource
                $.get("/arbiter/summary",function(data)"""),format.raw/*209.56*/("""{"""),format.raw/*209.57*/("""
                    """),format.raw/*210.21*/("""var summaryStatusDiv = $('#statusdiv');
                    summaryStatusDiv.html('');

                    var str = JSON.stringify(data);
                    var component = Component.getComponent(str);
                    component.render(summaryStatusDiv);
                """),format.raw/*216.17*/("""}"""),format.raw/*216.18*/(""");

                lastStatusUpdateTime = statusTime;
            """),format.raw/*219.13*/("""}"""),format.raw/*219.14*/("""

            """),format.raw/*221.13*/("""//Second section: Optimization settings
            if(lastSettingsUpdateTime != settingsTime)"""),format.raw/*222.55*/("""{"""),format.raw/*222.56*/("""
                """),format.raw/*223.17*/("""//Get JSON for components
                $.get("/arbiter/config",function(data)"""),format.raw/*224.55*/("""{"""),format.raw/*224.56*/("""
                    """),format.raw/*225.21*/("""var str = JSON.stringify(data);

                    var configDiv = $('#settingsdiv');
                    configDiv.html('');

                    var component = Component.getComponent(str);
                    component.render(configDiv);
                """),format.raw/*232.17*/("""}"""),format.raw/*232.18*/(""");

                lastSettingsUpdateTime = settingsTime;
            """),format.raw/*235.13*/("""}"""),format.raw/*235.14*/("""

            """),format.raw/*237.13*/("""//Third section: Summary results table (summary info for each candidate)
            if(lastResultsUpdateTime != resultsTime)"""),format.raw/*238.53*/("""{"""),format.raw/*238.54*/("""

                """),format.raw/*240.17*/("""//Get JSON; address set by SummaryResultsResource
                $.get("/arbiter/results",function(data)"""),format.raw/*241.56*/("""{"""),format.raw/*241.57*/("""
                    """),format.raw/*242.21*/("""//Expect an array of CandidateInfo type objects here
                    resultsTableContent = data;
                    drawResultTable();
                """),format.raw/*245.17*/("""}"""),format.raw/*245.18*/(""");

                lastResultsUpdateTime = resultsTime;
            """),format.raw/*248.13*/("""}"""),format.raw/*248.14*/("""

            """),format.raw/*250.13*/("""//Finally: Currently selected result
            if(selectedCandidateIdx != null)"""),format.raw/*251.45*/("""{"""),format.raw/*251.46*/("""
                """),format.raw/*252.17*/("""//Get JSON for components
                $.get("/arbiter/candidateInfo/"+selectedCandidateIdx,function(data)"""),format.raw/*253.84*/("""{"""),format.raw/*253.85*/("""
                    """),format.raw/*254.21*/("""var str = JSON.stringify(data);

                    var resultsViewDiv = $('#resultsviewdiv');
                    resultsViewDiv.html('');

                    var component = Component.getComponent(str);
                    component.render(resultsViewDiv);
                """),format.raw/*261.17*/("""}"""),format.raw/*261.18*/(""");
            """),format.raw/*262.13*/("""}"""),format.raw/*262.14*/("""
        """),format.raw/*263.9*/("""}"""),format.raw/*263.10*/(""")
    """),format.raw/*264.5*/("""}"""),format.raw/*264.6*/("""

    """),format.raw/*266.5*/("""function createTable(tableObj,tableId,appendTo)"""),format.raw/*266.52*/("""{"""),format.raw/*266.53*/("""
        """),format.raw/*267.9*/("""//Expect RenderableComponentTable
        var header = tableObj['header'];
        var values = tableObj['table'];
        var title = tableObj['title'];
        var nRows = (values ? values.length : 0);

        if(title)"""),format.raw/*273.18*/("""{"""),format.raw/*273.19*/("""
            """),format.raw/*274.13*/("""appendTo.append("<h5>"+title+"</h5>");
        """),format.raw/*275.9*/("""}"""),format.raw/*275.10*/("""

        """),format.raw/*277.9*/("""var table;
        if(tableId) table = $("<table id=\"" + tableId + "\" class=\"renderableComponentTable\">");
        else table = $("<table class=\"renderableComponentTable\">");
        if(header)"""),format.raw/*280.19*/("""{"""),format.raw/*280.20*/("""
            """),format.raw/*281.13*/("""var headerRow = $("<tr>");
            var len = header.length;
            for( var i=0; i<len; i++ )"""),format.raw/*283.39*/("""{"""),format.raw/*283.40*/("""
                """),format.raw/*284.17*/("""headerRow.append($("<th>" + header[i] + "</th>"));
            """),format.raw/*285.13*/("""}"""),format.raw/*285.14*/("""
            """),format.raw/*286.13*/("""headerRow.append($("</tr>"));
            table.append(headerRow);
        """),format.raw/*288.9*/("""}"""),format.raw/*288.10*/("""

        """),format.raw/*290.9*/("""if(values)"""),format.raw/*290.19*/("""{"""),format.raw/*290.20*/("""
            """),format.raw/*291.13*/("""for( var i=0; i<nRows; i++ )"""),format.raw/*291.41*/("""{"""),format.raw/*291.42*/("""
                """),format.raw/*292.17*/("""var row = $("<tr>");
                var rowValues = values[i];
                var len = rowValues.length;
                for( var j=0; j<len; j++ )"""),format.raw/*295.43*/("""{"""),format.raw/*295.44*/("""
                    """),format.raw/*296.21*/("""row.append($('<td>'+rowValues[j]+'</td>'));
                """),format.raw/*297.17*/("""}"""),format.raw/*297.18*/("""
                """),format.raw/*298.17*/("""row.append($("</tr>"));
                table.append(row);
            """),format.raw/*300.13*/("""}"""),format.raw/*300.14*/("""
        """),format.raw/*301.9*/("""}"""),format.raw/*301.10*/("""

        """),format.raw/*303.9*/("""table.append($("</table>"));
        appendTo.append(table);
    """),format.raw/*305.5*/("""}"""),format.raw/*305.6*/("""

    """),format.raw/*307.5*/("""function drawResultTable()"""),format.raw/*307.31*/("""{"""),format.raw/*307.32*/("""
        """),format.raw/*308.9*/("""//Remove all elements from the table body
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
        for(var i=0; i<len; i++)"""),format.raw/*333.33*/("""{"""),format.raw/*333.34*/("""
            """),format.raw/*334.13*/("""var row;
            if(selectedCandidateIdx == sorted[i][0])"""),format.raw/*335.53*/("""{"""),format.raw/*335.54*/("""
                """),format.raw/*336.17*/("""//Selected row
                row = $('<tr class="resultTableRowSelected" id="rTbl-' + sorted[i][0] + '"/>');
            """),format.raw/*338.13*/("""}"""),format.raw/*338.14*/(""" """),format.raw/*338.15*/("""else """),format.raw/*338.20*/("""{"""),format.raw/*338.21*/("""
                """),format.raw/*339.17*/("""//Normal row
                row = $('<tr class="resultTableRow" id="rTbl-' + sorted[i][0] + '"/>');
            """),format.raw/*341.13*/("""}"""),format.raw/*341.14*/("""
            """),format.raw/*342.13*/("""row.append($("<td>" + sorted[i][0] + "</td>"));
            var score = sorted[i][1];
            row.append($("<td>" + ((!score || score == "null") ? "-" : score) + "</td>"));
            row.append($("<td>" + sorted[i][2] + "</td>"));
            tableBody.append(row);
        """),format.raw/*347.9*/("""}"""),format.raw/*347.10*/("""
    """),format.raw/*348.5*/("""}"""),format.raw/*348.6*/("""

    """),format.raw/*350.5*/("""//Compare function for results, based on sort order
    function compareResultsIndex(a, b)"""),format.raw/*351.39*/("""{"""),format.raw/*351.40*/("""
        """),format.raw/*352.9*/("""return (resultTableSortOrder == "ascending" ? a[0] - b[0] : b[0] - a[0]);
    """),format.raw/*353.5*/("""}"""),format.raw/*353.6*/("""
    """),format.raw/*354.5*/("""function compareScores(a,b)"""),format.raw/*354.32*/("""{"""),format.raw/*354.33*/("""
        """),format.raw/*355.9*/("""//TODO Not always numbers...
        if(resultTableSortOrder == "ascending")"""),format.raw/*356.48*/("""{"""),format.raw/*356.49*/("""
            """),format.raw/*357.13*/("""if(a[1] == "NaN")"""),format.raw/*357.30*/("""{"""),format.raw/*357.31*/("""
                """),format.raw/*358.17*/("""return 1;
            """),format.raw/*359.13*/("""}"""),format.raw/*359.14*/(""" """),format.raw/*359.15*/("""else if(b[1] == "NaN")"""),format.raw/*359.37*/("""{"""),format.raw/*359.38*/("""
                """),format.raw/*360.17*/("""return -1;
            """),format.raw/*361.13*/("""}"""),format.raw/*361.14*/("""
            """),format.raw/*362.13*/("""return a[1] - b[1];
        """),format.raw/*363.9*/("""}"""),format.raw/*363.10*/(""" """),format.raw/*363.11*/("""else """),format.raw/*363.16*/("""{"""),format.raw/*363.17*/("""
            """),format.raw/*364.13*/("""if(a[1] == "NaN")"""),format.raw/*364.30*/("""{"""),format.raw/*364.31*/("""
                """),format.raw/*365.17*/("""return -1;
            """),format.raw/*366.13*/("""}"""),format.raw/*366.14*/(""" """),format.raw/*366.15*/("""else if(b[1] == "NaN")"""),format.raw/*366.37*/("""{"""),format.raw/*366.38*/("""
                """),format.raw/*367.17*/("""return 1;
            """),format.raw/*368.13*/("""}"""),format.raw/*368.14*/("""
            """),format.raw/*369.13*/("""return b[1] - a[1];
        """),format.raw/*370.9*/("""}"""),format.raw/*370.10*/("""
    """),format.raw/*371.5*/("""}"""),format.raw/*371.6*/("""
    """),format.raw/*372.5*/("""function compareStatus(a,b)"""),format.raw/*372.32*/("""{"""),format.raw/*372.33*/("""
        """),format.raw/*373.9*/("""//TODO: secondary sort on... score? index?
        if(resultTableSortOrder == "ascending")"""),format.raw/*374.48*/("""{"""),format.raw/*374.49*/("""
            """),format.raw/*375.13*/("""return (a[2] < b[2] ? -1 : (a[2] > b[2] ? 1 : 0));
        """),format.raw/*376.9*/("""}"""),format.raw/*376.10*/(""" """),format.raw/*376.11*/("""else """),format.raw/*376.16*/("""{"""),format.raw/*376.17*/("""
            """),format.raw/*377.13*/("""return (a[2] < b[2] ? 1 : (a[2] > b[2] ? -1 : 0));
        """),format.raw/*378.9*/("""}"""),format.raw/*378.10*/("""
    """),format.raw/*379.5*/("""}"""),format.raw/*379.6*/("""

    """),format.raw/*381.5*/("""//Do a HTTP request on the specified path, parse and insert into the provided element
    function loadCandidateDetails(path, elementToAppendTo)"""),format.raw/*382.59*/("""{"""),format.raw/*382.60*/("""
        """),format.raw/*383.9*/("""$.get(path, function (data) """),format.raw/*383.37*/("""{"""),format.raw/*383.38*/("""
            """),format.raw/*384.13*/("""var str = JSON.stringify(data);
            var component = Component.getComponent(str);
            component.render(elementToAppendTo);
        """),format.raw/*387.9*/("""}"""),format.raw/*387.10*/(""");
    """),format.raw/*388.5*/("""}"""),format.raw/*388.6*/("""



    """),format.raw/*392.5*/("""//Sorting by column: Intercept click events on table header
    $(function()"""),format.raw/*393.17*/("""{"""),format.raw/*393.18*/("""
        """),format.raw/*394.9*/("""$("#resultsTableHeader").delegate("th", "click", function(e) """),format.raw/*394.70*/("""{"""),format.raw/*394.71*/("""
            """),format.raw/*395.13*/("""//console.log("Header clicked on at: " + $(e.currentTarget).index() + " - " + $(e.currentTarget).html());
            //Update the sort order for the table:
            var clickIndex = $(e.currentTarget).index();
            if(clickIndex == resultTableSortIndex)"""),format.raw/*398.51*/("""{"""),format.raw/*398.52*/("""
                """),format.raw/*399.17*/("""//Switch sort order: ascending -> descending or descending -> ascending
                if(resultTableSortOrder == "ascending")"""),format.raw/*400.56*/("""{"""),format.raw/*400.57*/("""
                    """),format.raw/*401.21*/("""resultTableSortOrder = "descending";
                """),format.raw/*402.17*/("""}"""),format.raw/*402.18*/(""" """),format.raw/*402.19*/("""else """),format.raw/*402.24*/("""{"""),format.raw/*402.25*/("""
                    """),format.raw/*403.21*/("""resultTableSortOrder = "ascending";
                """),format.raw/*404.17*/("""}"""),format.raw/*404.18*/("""
            """),format.raw/*405.13*/("""}"""),format.raw/*405.14*/(""" """),format.raw/*405.15*/("""else """),format.raw/*405.20*/("""{"""),format.raw/*405.21*/("""
                """),format.raw/*406.17*/("""//Sort on column, ascending:
                resultTableSortIndex = clickIndex;
                resultTableSortOrder = "ascending";
            """),format.raw/*409.13*/("""}"""),format.raw/*409.14*/("""

            """),format.raw/*411.13*/("""//Clear record of expanded rows
            expandedRowsCandidateIDs = [];

            //Redraw table
            drawResultTable();
        """),format.raw/*416.9*/("""}"""),format.raw/*416.10*/(""");
    """),format.raw/*417.5*/("""}"""),format.raw/*417.6*/(""");

    //Displaying model/candidate details: Intercept click events on table rows -> toggle selected, fire off update
    $(function()"""),format.raw/*420.17*/("""{"""),format.raw/*420.18*/("""
        """),format.raw/*421.9*/("""$("#resultsTableBody").delegate("tr", "click", function(e)"""),format.raw/*421.67*/("""{"""),format.raw/*421.68*/("""
            """),format.raw/*422.13*/("""var id = this.id;   //Expect: rTbl-X  where X is some index
            var dashIdx = id.indexOf("-");
            var candidateID = Number(id.substring(dashIdx+1));
//            console.log("Clicked row: " + this.id + " with class: " + this.className + ", candidateId = " + candidateID);

            if(this.className == "resultTableRow")"""),format.raw/*427.51*/("""{"""),format.raw/*427.52*/("""
                """),format.raw/*428.17*/("""//Set selected model
                selectedCandidateIdx = candidateID;

                //Fire off update
                doUpdate();
            """),format.raw/*433.13*/("""}"""),format.raw/*433.14*/("""
        """),format.raw/*434.9*/("""}"""),format.raw/*434.10*/(""");
    """),format.raw/*435.5*/("""}"""),format.raw/*435.6*/(""");

</script>
<script>
    $(function () """),format.raw/*439.19*/("""{"""),format.raw/*439.20*/("""
        """),format.raw/*440.9*/("""$("#accordion").accordion("""),format.raw/*440.35*/("""{"""),format.raw/*440.36*/("""
            """),format.raw/*441.13*/("""collapsible: true,
            heightStyle: "content"
        """),format.raw/*443.9*/("""}"""),format.raw/*443.10*/(""");
    """),format.raw/*444.5*/("""}"""),format.raw/*444.6*/(""");
    $(function () """),format.raw/*445.19*/("""{"""),format.raw/*445.20*/("""
        """),format.raw/*446.9*/("""$("#accordion2").accordion("""),format.raw/*446.36*/("""{"""),format.raw/*446.37*/("""
            """),format.raw/*447.13*/("""collapsible: true,
            heightStyle: "content"
        """),format.raw/*449.9*/("""}"""),format.raw/*449.10*/(""");
    """),format.raw/*450.5*/("""}"""),format.raw/*450.6*/(""");
    $(function () """),format.raw/*451.19*/("""{"""),format.raw/*451.20*/("""
        """),format.raw/*452.9*/("""$("#accordion3").accordion("""),format.raw/*452.36*/("""{"""),format.raw/*452.37*/("""
            """),format.raw/*453.13*/("""collapsible: true,
            heightStyle: "content"
        """),format.raw/*455.9*/("""}"""),format.raw/*455.10*/(""");
    """),format.raw/*456.5*/("""}"""),format.raw/*456.6*/(""");
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
                  DATE: Fri Jul 21 22:22:19 AEST 2017
                  SOURCE: C:/DL4J/Git/Arbiter/arbiter-ui/src/main/views/org/deeplearning4j/arbiter/ui/views/ArbiterUI.scala.html
                  HASH: 34e77bad09019b2f9462f218fdd77416d016b8e5
                  MATRIX: 647->0|1029->354|1058->355|1100->369|1331->573|1360->574|1399->586|1438->597|1467->598|1509->612|1610->686|1639->687|1678->699|1715->708|1744->709|1786->723|1849->759|1878->760|1917->772|1948->775|1977->776|2019->790|2254->998|2283->999|2322->1011|2353->1014|2382->1015|2424->1029|2661->1239|2690->1240|2729->1252|2763->1258|2792->1259|2834->1273|2938->1350|2967->1351|3006->1363|3053->1382|3082->1383|3124->1397|3296->1542|3325->1543|3364->1555|3460->1623|3489->1624|3531->1638|3673->1753|3702->1754|3741->1766|3791->1788|3820->1789|3862->1803|4033->1947|4062->1948|4101->1960|4151->1982|4180->1983|4222->1997|4352->2100|4381->2101|4420->2113|4580->2245|4609->2246|4651->2260|4804->2386|4833->2387|4870->2397|4927->2426|4956->2427|4998->2441|5254->2670|5283->2671|5322->2683|5408->2741|5437->2742|5479->2756|5572->2822|5601->2823|5640->2835|5777->2944|5806->2945|5848->2959|5928->3011|5958->3012|5998->3024|6046->3043|6076->3044|6119->3058|6702->3613|6732->3614|6772->3626|6819->3644|6849->3645|6892->3659|6951->3690|6981->3691|7021->3703|7074->3727|7104->3728|7147->3742|7206->3773|7236->3774|7276->3786|7406->3886|7437->3887|7480->3901|7918->4311|7948->4312|7988->4324|8050->4357|8080->4358|8123->4372|8454->4675|8484->4676|8524->4688|8586->4721|8616->4722|8659->4736|8770->4819|8800->4820|8838->4830|8890->4853|8920->4854|8963->4868|9081->4958|9111->4959|9149->4969|9189->4980|9219->4981|9262->4995|9355->5060|9385->5061|9425->5073|10606->6225|10636->6226|10674->6236|10805->6338|10835->6339|10878->6353|11494->6940|11524->6941|11571->6959|11705->7064|11735->7065|11786->7087|12098->7370|12128->7371|12227->7441|12257->7442|12302->7458|12426->7553|12456->7554|12503->7572|12613->7653|12643->7654|12694->7676|12989->7942|13019->7943|13122->8017|13152->8018|13197->8034|13352->8160|13382->8161|13431->8181|13566->8287|13596->8288|13647->8310|13835->8469|13865->8470|13966->8542|13996->8543|14041->8559|14152->8641|14182->8642|14229->8660|14368->8770|14398->8771|14449->8793|14762->9077|14792->9078|14837->9094|14867->9095|14905->9105|14935->9106|14970->9113|14999->9114|15035->9122|15111->9169|15141->9170|15179->9180|15436->9408|15466->9409|15509->9423|15585->9471|15615->9472|15655->9484|15886->9686|15916->9687|15959->9701|16092->9805|16122->9806|16169->9824|16262->9888|16292->9889|16335->9903|16440->9980|16470->9981|16510->9993|16549->10003|16579->10004|16622->10018|16679->10046|16709->10047|16756->10065|16938->10218|16968->10219|17019->10241|17109->10302|17139->10303|17186->10321|17288->10394|17318->10395|17356->10405|17386->10406|17426->10418|17521->10485|17550->10486|17586->10494|17641->10520|17671->10521|17709->10531|19066->11859|19096->11860|19139->11874|19230->11936|19260->11937|19307->11955|19461->12080|19491->12081|19521->12082|19555->12087|19585->12088|19632->12106|19776->12221|19806->12222|19849->12236|20162->12521|20192->12522|20226->12528|20255->12529|20291->12537|20411->12628|20441->12629|20479->12639|20586->12718|20615->12719|20649->12725|20705->12752|20735->12753|20773->12763|20879->12840|20909->12841|20952->12855|20998->12872|21028->12873|21075->12891|21127->12914|21157->12915|21187->12916|21238->12938|21268->12939|21315->12957|21368->12981|21398->12982|21441->12996|21498->13025|21528->13026|21558->13027|21592->13032|21622->13033|21665->13047|21711->13064|21741->13065|21788->13083|21841->13107|21871->13108|21901->13109|21952->13131|21982->13132|22029->13150|22081->13173|22111->13174|22154->13188|22211->13217|22241->13218|22275->13224|22304->13225|22338->13231|22394->13258|22424->13259|22462->13269|22582->13360|22612->13361|22655->13375|22743->13435|22773->13436|22803->13437|22837->13442|22867->13443|22910->13457|22998->13517|23028->13518|23062->13524|23091->13525|23127->13533|23301->13678|23331->13679|23369->13689|23426->13717|23456->13718|23499->13732|23676->13881|23706->13882|23742->13890|23771->13891|23811->13903|23917->13980|23947->13981|23985->13991|24075->14052|24105->14053|24148->14067|24444->14334|24474->14335|24521->14353|24678->14481|24708->14482|24759->14504|24842->14558|24872->14559|24902->14560|24936->14565|24966->14566|25017->14588|25099->14641|25129->14642|25172->14656|25202->14657|25232->14658|25266->14663|25296->14664|25343->14682|25519->14829|25549->14830|25594->14846|25769->14993|25799->14994|25835->15002|25864->15003|26031->15141|26061->15142|26099->15152|26186->15210|26216->15211|26259->15225|26634->15571|26664->15572|26711->15590|26893->15743|26923->15744|26961->15754|26991->15755|27027->15763|27056->15764|27130->15809|27160->15810|27198->15820|27253->15846|27283->15847|27326->15861|27418->15925|27448->15926|27484->15934|27513->15935|27564->15957|27594->15958|27632->15968|27688->15995|27718->15996|27761->16010|27853->16074|27883->16075|27919->16083|27948->16084|27999->16106|28029->16107|28067->16117|28123->16144|28153->16145|28196->16159|28288->16223|28318->16224|28354->16232|28383->16233
                  LINES: 25->1|35->11|35->11|36->12|42->18|42->18|44->20|44->20|44->20|45->21|48->24|48->24|50->26|50->26|50->26|51->27|52->28|52->28|54->30|54->30|54->30|55->31|61->37|61->37|63->39|63->39|63->39|64->40|70->46|70->46|72->48|72->48|72->48|73->49|75->51|75->51|77->53|77->53|77->53|78->54|82->58|82->58|84->60|84->60|84->60|85->61|87->63|87->63|89->65|89->65|89->65|90->66|94->70|94->70|96->72|96->72|96->72|97->73|100->76|100->76|102->78|103->79|103->79|104->80|106->82|106->82|107->83|107->83|107->83|108->84|113->89|113->89|115->91|116->92|116->92|117->93|119->95|119->95|121->97|122->98|122->98|123->99|124->100|124->100|126->102|126->102|126->102|127->103|143->119|143->119|145->121|145->121|145->121|146->122|147->123|147->123|149->125|149->125|149->125|150->126|151->127|151->127|153->129|153->129|153->129|154->130|162->138|162->138|164->140|164->140|164->140|165->141|171->147|171->147|173->149|174->150|174->150|175->151|178->154|178->154|179->155|179->155|179->155|180->156|183->159|183->159|184->160|184->160|184->160|185->161|187->163|187->163|189->165|219->195|219->195|220->196|221->197|221->197|222->198|231->207|231->207|232->208|233->209|233->209|234->210|240->216|240->216|243->219|243->219|245->221|246->222|246->222|247->223|248->224|248->224|249->225|256->232|256->232|259->235|259->235|261->237|262->238|262->238|264->240|265->241|265->241|266->242|269->245|269->245|272->248|272->248|274->250|275->251|275->251|276->252|277->253|277->253|278->254|285->261|285->261|286->262|286->262|287->263|287->263|288->264|288->264|290->266|290->266|290->266|291->267|297->273|297->273|298->274|299->275|299->275|301->277|304->280|304->280|305->281|307->283|307->283|308->284|309->285|309->285|310->286|312->288|312->288|314->290|314->290|314->290|315->291|315->291|315->291|316->292|319->295|319->295|320->296|321->297|321->297|322->298|324->300|324->300|325->301|325->301|327->303|329->305|329->305|331->307|331->307|331->307|332->308|357->333|357->333|358->334|359->335|359->335|360->336|362->338|362->338|362->338|362->338|362->338|363->339|365->341|365->341|366->342|371->347|371->347|372->348|372->348|374->350|375->351|375->351|376->352|377->353|377->353|378->354|378->354|378->354|379->355|380->356|380->356|381->357|381->357|381->357|382->358|383->359|383->359|383->359|383->359|383->359|384->360|385->361|385->361|386->362|387->363|387->363|387->363|387->363|387->363|388->364|388->364|388->364|389->365|390->366|390->366|390->366|390->366|390->366|391->367|392->368|392->368|393->369|394->370|394->370|395->371|395->371|396->372|396->372|396->372|397->373|398->374|398->374|399->375|400->376|400->376|400->376|400->376|400->376|401->377|402->378|402->378|403->379|403->379|405->381|406->382|406->382|407->383|407->383|407->383|408->384|411->387|411->387|412->388|412->388|416->392|417->393|417->393|418->394|418->394|418->394|419->395|422->398|422->398|423->399|424->400|424->400|425->401|426->402|426->402|426->402|426->402|426->402|427->403|428->404|428->404|429->405|429->405|429->405|429->405|429->405|430->406|433->409|433->409|435->411|440->416|440->416|441->417|441->417|444->420|444->420|445->421|445->421|445->421|446->422|451->427|451->427|452->428|457->433|457->433|458->434|458->434|459->435|459->435|463->439|463->439|464->440|464->440|464->440|465->441|467->443|467->443|468->444|468->444|469->445|469->445|470->446|470->446|470->446|471->447|473->449|473->449|474->450|474->450|475->451|475->451|476->452|476->452|476->452|477->453|479->455|479->455|480->456|480->456
                  -- GENERATED --
              */
          