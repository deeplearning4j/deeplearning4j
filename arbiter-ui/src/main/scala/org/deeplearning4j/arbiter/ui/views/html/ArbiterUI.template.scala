
package org.deeplearning4j.arbiter.ui.views.html

import play.twirl.api._


     object ArbiterUI_Scope0 {

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
        .resultTableRowContent """),format.raw/*91.32*/("""{"""),format.raw/*91.33*/("""
            """),format.raw/*92.13*/("""background-color: white;
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

        """),format.raw/*133.9*/("""/*
        #accordion .ui-accordion-header.ui-state-active """),format.raw/*134.57*/("""{"""),format.raw/*134.58*/("""
            """),format.raw/*135.13*/("""background-color: pink;
            background-image: none;
        """),format.raw/*137.9*/("""}"""),format.raw/*137.10*/("""*/

        #accordion .ui-accordion-content """),format.raw/*139.42*/("""{"""),format.raw/*139.43*/("""
            """),format.raw/*140.13*/("""width: 100%;
            background-color: white;    /*background color of accordian content (elements in front may have different color */
            color: black;  /* text etc color */
            font-size: 10pt;
            line-height: 16pt;
            overflow:visible !important;
        """),format.raw/*146.9*/("""}"""),format.raw/*146.10*/("""

        """),format.raw/*148.9*/("""/** Line charts */
        path """),format.raw/*149.14*/("""{"""),format.raw/*149.15*/("""
            """),format.raw/*150.13*/("""stroke: steelblue;
            stroke-width: 2;
            fill: none;
        """),format.raw/*153.9*/("""}"""),format.raw/*153.10*/("""
        """),format.raw/*154.9*/(""".axis path, .axis line """),format.raw/*154.32*/("""{"""),format.raw/*154.33*/("""
            """),format.raw/*155.13*/("""fill: none;
            stroke: #000;
            shape-rendering: crispEdges;
        """),format.raw/*158.9*/("""}"""),format.raw/*158.10*/("""
        """),format.raw/*159.9*/(""".tick line """),format.raw/*159.20*/("""{"""),format.raw/*159.21*/("""
            """),format.raw/*160.13*/("""opacity: 0.2;
            shape-rendering: crispEdges;
        """),format.raw/*162.9*/("""}"""),format.raw/*162.10*/("""

        """),format.raw/*164.9*/("""</style>
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

    var expandedRowsCandidateIDs = [];

    //Set basic interval function to do updates
    setInterval(function()"""),format.raw/*193.27*/("""{"""),format.raw/*193.28*/("""
        """),format.raw/*194.9*/("""//Get the update status, and do something with it:
        $.get("/arbiter/lastUpdate",function(data)"""),format.raw/*195.51*/("""{"""),format.raw/*195.52*/("""
            """),format.raw/*196.13*/("""//Encoding: matches names in UpdateStatus class
            var jsonObj = JSON.parse(JSON.stringify(data));
            var statusTime = jsonObj['statusUpdateTime'];
            var settingsTime = jsonObj['settingsUpdateTime'];
            var resultsTime = jsonObj['resultsUpdateTime'];
            //console.log("Last update times: " + statusTime + ", " + settingsTime + ", " + resultsTime);

            //Check last update times for each part of document, and update as necessary
            //First section: summary status
            if(lastStatusUpdateTime != statusTime)"""),format.raw/*205.51*/("""{"""),format.raw/*205.52*/("""
                """),format.raw/*206.17*/("""//Get JSON: address set by SummaryStatusResource
                $.get("/arbiter/summary",function(data)"""),format.raw/*207.56*/("""{"""),format.raw/*207.57*/("""
                    """),format.raw/*208.21*/("""var summaryStatusDiv = $('#statusdiv');
                    summaryStatusDiv.html('');

                    var str = JSON.stringify(data);
                    var component = Component.getComponent(str);
                    component.render(summaryStatusDiv);
                """),format.raw/*214.17*/("""}"""),format.raw/*214.18*/(""");

                lastStatusUpdateTime = statusTime;
            """),format.raw/*217.13*/("""}"""),format.raw/*217.14*/("""

            """),format.raw/*219.13*/("""//Second section: Optimization settings
            if(lastSettingsUpdateTime != settingsTime)"""),format.raw/*220.55*/("""{"""),format.raw/*220.56*/("""
                """),format.raw/*221.17*/("""//Get JSON: address set by ConfigResource
                $.get("/arbiter/config",function(data)"""),format.raw/*222.55*/("""{"""),format.raw/*222.56*/("""
                    """),format.raw/*223.21*/("""var str = JSON.stringify(data);

                    var configDiv = $('#settingsdiv');
                    configDiv.html('');

                    var component = Component.getComponent(str);
                    component.render(configDiv);
                """),format.raw/*230.17*/("""}"""),format.raw/*230.18*/(""");

                lastSettingsUpdateTime = settingsTime;
            """),format.raw/*233.13*/("""}"""),format.raw/*233.14*/("""

            """),format.raw/*235.13*/("""//Third section: Summary results table (summary info for each candidate)
            if(lastResultsUpdateTime != resultsTime)"""),format.raw/*236.53*/("""{"""),format.raw/*236.54*/("""

                """),format.raw/*238.17*/("""//Get JSON; address set by SummaryResultsResource
                $.get("/arbiter/results",function(data)"""),format.raw/*239.56*/("""{"""),format.raw/*239.57*/("""
                    """),format.raw/*240.21*/("""//Expect an array of CandidateInfo type objects here
                    resultsTableContent = data;
                    drawResultTable();
                """),format.raw/*243.17*/("""}"""),format.raw/*243.18*/(""");

                lastResultsUpdateTime = resultsTime;
            """),format.raw/*246.13*/("""}"""),format.raw/*246.14*/("""
        """),format.raw/*247.9*/("""}"""),format.raw/*247.10*/(""")
    """),format.raw/*248.5*/("""}"""),format.raw/*248.6*/(""",2000);    //Loop every 2 seconds

    function createTable(tableObj,tableId,appendTo)"""),format.raw/*250.52*/("""{"""),format.raw/*250.53*/("""
        """),format.raw/*251.9*/("""//Expect RenderableComponentTable
        var header = tableObj['header'];
        var values = tableObj['table'];
        var title = tableObj['title'];
        var nRows = (values ? values.length : 0);

        if(title)"""),format.raw/*257.18*/("""{"""),format.raw/*257.19*/("""
            """),format.raw/*258.13*/("""appendTo.append("<h5>"+title+"</h5>");
        """),format.raw/*259.9*/("""}"""),format.raw/*259.10*/("""

        """),format.raw/*261.9*/("""var table;
        if(tableId) table = $("<table id=\"" + tableId + "\" class=\"renderableComponentTable\">");
        else table = $("<table class=\"renderableComponentTable\">");
        if(header)"""),format.raw/*264.19*/("""{"""),format.raw/*264.20*/("""
            """),format.raw/*265.13*/("""var headerRow = $("<tr>");
            var len = header.length;
            for( var i=0; i<len; i++ )"""),format.raw/*267.39*/("""{"""),format.raw/*267.40*/("""
                """),format.raw/*268.17*/("""headerRow.append($("<th>" + header[i] + "</th>"));
            """),format.raw/*269.13*/("""}"""),format.raw/*269.14*/("""
            """),format.raw/*270.13*/("""headerRow.append($("</tr>"));
            table.append(headerRow);
        """),format.raw/*272.9*/("""}"""),format.raw/*272.10*/("""

        """),format.raw/*274.9*/("""if(values)"""),format.raw/*274.19*/("""{"""),format.raw/*274.20*/("""
            """),format.raw/*275.13*/("""for( var i=0; i<nRows; i++ )"""),format.raw/*275.41*/("""{"""),format.raw/*275.42*/("""
                """),format.raw/*276.17*/("""var row = $("<tr>");
                var rowValues = values[i];
                var len = rowValues.length;
                for( var j=0; j<len; j++ )"""),format.raw/*279.43*/("""{"""),format.raw/*279.44*/("""
                    """),format.raw/*280.21*/("""row.append($('<td>'+rowValues[j]+'</td>'));
                """),format.raw/*281.17*/("""}"""),format.raw/*281.18*/("""
                """),format.raw/*282.17*/("""row.append($("</tr>"));
                table.append(row);
            """),format.raw/*284.13*/("""}"""),format.raw/*284.14*/("""
        """),format.raw/*285.9*/("""}"""),format.raw/*285.10*/("""

        """),format.raw/*287.9*/("""table.append($("</table>"));
        appendTo.append(table);
    """),format.raw/*289.5*/("""}"""),format.raw/*289.6*/("""

    """),format.raw/*291.5*/("""function drawResultTable()"""),format.raw/*291.31*/("""{"""),format.raw/*291.32*/("""

        """),format.raw/*293.9*/("""//Remove all elements from the table body
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
        for(var i=0; i<len; i++)"""),format.raw/*318.33*/("""{"""),format.raw/*318.34*/("""
            """),format.raw/*319.13*/("""var row = $('<tr class="resultTableRow" id="resultTableRow-' + sorted[i].index + '"/>');
            row.append($("<td>" + sorted[i][0] + "</td>"));
            var score = sorted[i][1];
            row.append($("<td>" + ((!score || score == "null") ? "-" : score) + "</td>"));
            row.append($("<td>" + sorted[i][2] + "</td>"));
            tableBody.append(row);

            //Create hidden row for expanding:
            var rowID = 'resultTableRow-' + sorted[i][0] + '-content';
            var contentRow = $('<tr id="' + rowID + '" class="resultTableRowContent"/>');
            var td3 = $("<td colspan=3 id=" + rowID + "-td></td>");
            td3.append("(Result status - loading)");
            contentRow.append(td3);

            tableBody.append(contentRow);
            if(expandedRowsCandidateIDs.indexOf(sorted[i][0]) == -1 )"""),format.raw/*334.70*/("""{"""),format.raw/*334.71*/("""
                """),format.raw/*335.17*/("""contentRow.hide();

            """),format.raw/*337.13*/("""}"""),format.raw/*337.14*/(""" """),format.raw/*337.15*/("""else """),format.raw/*337.20*/("""{"""),format.raw/*337.21*/("""
                """),format.raw/*338.17*/("""//Load info. TODO: make this more efficient (stored info, check for updates, etc)
                td3.empty();

                var path = "/modelResults/" + sorted[i][0];
                loadCandidateDetails(path, td3);

                contentRow.show();
            """),format.raw/*345.13*/("""}"""),format.raw/*345.14*/("""
        """),format.raw/*346.9*/("""}"""),format.raw/*346.10*/("""
    """),format.raw/*347.5*/("""}"""),format.raw/*347.6*/("""

    """),format.raw/*349.5*/("""//Compare function for results, based on sort order
    function compareResultsIndex(a, b)"""),format.raw/*350.39*/("""{"""),format.raw/*350.40*/("""
        """),format.raw/*351.9*/("""return (resultTableSortOrder == "ascending" ? a[0] - b[0] : b[0] - a[0]);
    """),format.raw/*352.5*/("""}"""),format.raw/*352.6*/("""
    """),format.raw/*353.5*/("""function compareScores(a,b)"""),format.raw/*353.32*/("""{"""),format.raw/*353.33*/("""
        """),format.raw/*354.9*/("""//TODO Not always numbers...
        if(resultTableSortOrder == "ascending")"""),format.raw/*355.48*/("""{"""),format.raw/*355.49*/("""
            """),format.raw/*356.13*/("""return a[1] - b[1];
        """),format.raw/*357.9*/("""}"""),format.raw/*357.10*/(""" """),format.raw/*357.11*/("""else """),format.raw/*357.16*/("""{"""),format.raw/*357.17*/("""
            """),format.raw/*358.13*/("""return b[1] - a[1];
        """),format.raw/*359.9*/("""}"""),format.raw/*359.10*/("""
    """),format.raw/*360.5*/("""}"""),format.raw/*360.6*/("""
    """),format.raw/*361.5*/("""function compareStatus(a,b)"""),format.raw/*361.32*/("""{"""),format.raw/*361.33*/("""
        """),format.raw/*362.9*/("""//TODO: secondary sort on... score? index?
        if(resultTableSortOrder == "ascending")"""),format.raw/*363.48*/("""{"""),format.raw/*363.49*/("""
            """),format.raw/*364.13*/("""return (a[2] < b[2] ? -1 : (a[2] > b[2] ? 1 : 0));
        """),format.raw/*365.9*/("""}"""),format.raw/*365.10*/(""" """),format.raw/*365.11*/("""else """),format.raw/*365.16*/("""{"""),format.raw/*365.17*/("""
            """),format.raw/*366.13*/("""return (a[2] < b[2] ? 1 : (a[2] > b[2] ? -1 : 0));
        """),format.raw/*367.9*/("""}"""),format.raw/*367.10*/("""
    """),format.raw/*368.5*/("""}"""),format.raw/*368.6*/("""

    """),format.raw/*370.5*/("""//Do a HTTP request on the specified path, parse and insert into the provided element
    function loadCandidateDetails(path, elementToAppendTo)"""),format.raw/*371.59*/("""{"""),format.raw/*371.60*/("""
        """),format.raw/*372.9*/("""$.get(path, function (data) """),format.raw/*372.37*/("""{"""),format.raw/*372.38*/("""
            """),format.raw/*373.13*/("""var str = JSON.stringify(data);
            var component = Component.getComponent(str);
            component.render(elementToAppendTo);
        """),format.raw/*376.9*/("""}"""),format.raw/*376.10*/(""");
    """),format.raw/*377.5*/("""}"""),format.raw/*377.6*/("""



    """),format.raw/*381.5*/("""//Sorting by column: Intercept click events on table header
    $(function()"""),format.raw/*382.17*/("""{"""),format.raw/*382.18*/("""
        """),format.raw/*383.9*/("""$("#resultsTableHeader").delegate("th", "click", function(e) """),format.raw/*383.70*/("""{"""),format.raw/*383.71*/("""
            """),format.raw/*384.13*/("""//console.log("Header clicked on at: " + $(e.currentTarget).index() + " - " + $(e.currentTarget).html());
            //Update the sort order for the table:
            var clickIndex = $(e.currentTarget).index();
            if(clickIndex == resultTableSortIndex)"""),format.raw/*387.51*/("""{"""),format.raw/*387.52*/("""
                """),format.raw/*388.17*/("""//Switch sort order: ascending -> descending or descending -> ascending
                if(resultTableSortOrder == "ascending")"""),format.raw/*389.56*/("""{"""),format.raw/*389.57*/("""
                    """),format.raw/*390.21*/("""resultTableSortOrder = "descending";
                """),format.raw/*391.17*/("""}"""),format.raw/*391.18*/(""" """),format.raw/*391.19*/("""else """),format.raw/*391.24*/("""{"""),format.raw/*391.25*/("""
                    """),format.raw/*392.21*/("""resultTableSortOrder = "ascending";
                """),format.raw/*393.17*/("""}"""),format.raw/*393.18*/("""
            """),format.raw/*394.13*/("""}"""),format.raw/*394.14*/(""" """),format.raw/*394.15*/("""else """),format.raw/*394.20*/("""{"""),format.raw/*394.21*/("""
                """),format.raw/*395.17*/("""//Sort on column, ascending:
                resultTableSortIndex = clickIndex;
                resultTableSortOrder = "ascending";
            """),format.raw/*398.13*/("""}"""),format.raw/*398.14*/("""

            """),format.raw/*400.13*/("""//Clear record of expanded rows
            expandedRowsCandidateIDs = [];

            //Redraw table
            drawResultTable();
        """),format.raw/*405.9*/("""}"""),format.raw/*405.10*/(""");
    """),format.raw/*406.5*/("""}"""),format.raw/*406.6*/(""");

    //Displaying model/candidate details: Intercept click events on table rows -> toggle visibility on content rows
    $(function()"""),format.raw/*409.17*/("""{"""),format.raw/*409.18*/("""
        """),format.raw/*410.9*/("""$("#resultsTableBody").delegate("tr", "click", function(e)"""),format.raw/*410.67*/("""{"""),format.raw/*410.68*/("""
"""),format.raw/*411.1*/("""//            console.log("Clicked row: " + this.id + " with class: " + this.className);
            var id = this.id;   //Expect: resultTableRow-X  where X is some index
            var dashIdx = id.indexOf("-");
            var candidateID = Number(id.substring(dashIdx+1));
            if(this.className == "resultTableRow")"""),format.raw/*415.51*/("""{"""),format.raw/*415.52*/("""
                """),format.raw/*416.17*/("""var contentRow = $('#' + this.id + '-content');
                var expRowsArrayIdx = expandedRowsCandidateIDs.indexOf(candidateID);
                if(expRowsArrayIdx == -1 )"""),format.raw/*418.43*/("""{"""),format.raw/*418.44*/("""
                    """),format.raw/*419.21*/("""//Currently hidden
                    expandedRowsCandidateIDs.push(candidateID); //Mark as expanded
                    var innerTD = $('#' + this.id + '-content-td');
                    innerTD.empty();
                    var path = "/modelResults/" + candidateID;
                    loadCandidateDetails(path,innerTD);
                """),format.raw/*425.17*/("""}"""),format.raw/*425.18*/(""" """),format.raw/*425.19*/("""else """),format.raw/*425.24*/("""{"""),format.raw/*425.25*/("""
                    """),format.raw/*426.21*/("""//Currently expanded
                    expandedRowsCandidateIDs.splice(expRowsArrayIdx,1);
                """),format.raw/*428.17*/("""}"""),format.raw/*428.18*/("""
                """),format.raw/*429.17*/("""contentRow.toggle();
            """),format.raw/*430.13*/("""}"""),format.raw/*430.14*/("""
        """),format.raw/*431.9*/("""}"""),format.raw/*431.10*/(""");
    """),format.raw/*432.5*/("""}"""),format.raw/*432.6*/(""");

</script>
<script>
    $(function () """),format.raw/*436.19*/("""{"""),format.raw/*436.20*/("""
        """),format.raw/*437.9*/("""$("#accordion").accordion("""),format.raw/*437.35*/("""{"""),format.raw/*437.36*/("""
            """),format.raw/*438.13*/("""collapsible: true,
            heightStyle: "content"
        """),format.raw/*440.9*/("""}"""),format.raw/*440.10*/(""");
    """),format.raw/*441.5*/("""}"""),format.raw/*441.6*/(""");
    $(function () """),format.raw/*442.19*/("""{"""),format.raw/*442.20*/("""
        """),format.raw/*443.9*/("""$("#accordion2").accordion("""),format.raw/*443.36*/("""{"""),format.raw/*443.37*/("""
            """),format.raw/*444.13*/("""collapsible: true,
            heightStyle: "content"
        """),format.raw/*446.9*/("""}"""),format.raw/*446.10*/(""");
    """),format.raw/*447.5*/("""}"""),format.raw/*447.6*/(""");
    $(function () """),format.raw/*448.19*/("""{"""),format.raw/*448.20*/("""
        """),format.raw/*449.9*/("""$("#accordion3").accordion("""),format.raw/*449.36*/("""{"""),format.raw/*449.37*/("""
            """),format.raw/*450.13*/("""collapsible: true,
            heightStyle: "content"
        """),format.raw/*452.9*/("""}"""),format.raw/*452.10*/(""");
    """),format.raw/*453.5*/("""}"""),format.raw/*453.6*/(""");
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
                  DATE: Wed Jul 19 21:58:09 AEST 2017
                  SOURCE: C:/DL4J/Git/Arbiter/arbiter-ui/src/main/views/org/deeplearning4j/arbiter/ui/views/ArbiterUI.scala.html
                  HASH: f08a1b788972cb29d2adfbc031555b267202aefa
                  MATRIX: 647->0|1031->356|1060->357|1102->371|1224->466|1253->467|1292->479|1331->490|1360->491|1402->505|1503->579|1532->580|1571->592|1608->601|1637->602|1679->616|1742->652|1771->653|1810->665|1841->668|1870->669|1912->683|2164->908|2193->909|2232->921|2263->924|2292->925|2334->939|2588->1166|2617->1167|2656->1179|2703->1198|2732->1199|2774->1213|2946->1358|2975->1359|3014->1371|3110->1439|3139->1440|3181->1454|3323->1569|3352->1570|3391->1582|3441->1604|3470->1605|3512->1619|3683->1763|3712->1764|3751->1776|3801->1798|3830->1799|3872->1813|4002->1916|4031->1917|4070->1929|4230->2061|4259->2062|4301->2076|4454->2202|4483->2203|4520->2213|4577->2242|4606->2243|4648->2257|4904->2486|4933->2487|4972->2499|5058->2557|5087->2558|5129->2572|5222->2638|5251->2639|5290->2651|5426->2759|5455->2760|5497->2774|5558->2808|5587->2809|5626->2821|5673->2840|5702->2841|5744->2855|6344->3427|6374->3428|6414->3440|6461->3458|6491->3459|6534->3473|6593->3504|6623->3505|6663->3517|6716->3541|6746->3542|6789->3556|6848->3587|6878->3588|6918->3600|7048->3700|7079->3701|7122->3715|7577->4142|7607->4143|7647->4155|7736->4215|7766->4216|7809->4230|7907->4300|7937->4301|8013->4348|8043->4349|8086->4363|8417->4666|8447->4667|8487->4679|8549->4712|8579->4713|8622->4727|8733->4810|8763->4811|8801->4821|8853->4844|8883->4845|8926->4859|9044->4949|9074->4950|9112->4960|9152->4971|9182->4972|9225->4986|9318->5051|9348->5052|9388->5064|10661->6308|10691->6309|10729->6319|10860->6421|10890->6422|10933->6436|11549->7023|11579->7024|11626->7042|11760->7147|11790->7148|11841->7170|12153->7453|12183->7454|12282->7524|12312->7525|12357->7541|12481->7636|12511->7637|12558->7655|12684->7752|12714->7753|12765->7775|13060->8041|13090->8042|13193->8116|13223->8117|13268->8133|13423->8259|13453->8260|13502->8280|13637->8386|13667->8387|13718->8409|13908->8570|13938->8571|14039->8643|14069->8644|14107->8654|14137->8655|14172->8662|14201->8663|14318->8751|14348->8752|14386->8762|14643->8990|14673->8991|14716->9005|14792->9053|14822->9054|14862->9066|15093->9268|15123->9269|15166->9283|15299->9387|15329->9388|15376->9406|15469->9470|15499->9471|15542->9485|15647->9562|15677->9563|15717->9575|15756->9585|15786->9586|15829->9600|15886->9628|15916->9629|15963->9647|16145->9800|16175->9801|16226->9823|16316->9884|16346->9885|16393->9903|16495->9976|16525->9977|16563->9987|16593->9988|16633->10000|16728->10067|16757->10068|16793->10076|16848->10102|16878->10103|16918->10115|18275->11443|18305->11444|18348->11458|19243->12324|19273->12325|19320->12343|19383->12377|19413->12378|19443->12379|19477->12384|19507->12385|19554->12403|19859->12679|19889->12680|19927->12690|19957->12691|19991->12697|20020->12698|20056->12706|20176->12797|20206->12798|20244->12808|20351->12887|20380->12888|20414->12894|20470->12921|20500->12922|20538->12932|20644->13009|20674->13010|20717->13024|20774->13053|20804->13054|20834->13055|20868->13060|20898->13061|20941->13075|20998->13104|21028->13105|21062->13111|21091->13112|21125->13118|21181->13145|21211->13146|21249->13156|21369->13247|21399->13248|21442->13262|21530->13322|21560->13323|21590->13324|21624->13329|21654->13330|21697->13344|21785->13404|21815->13405|21849->13411|21878->13412|21914->13420|22088->13565|22118->13566|22156->13576|22213->13604|22243->13605|22286->13619|22463->13768|22493->13769|22529->13777|22558->13778|22598->13790|22704->13867|22734->13868|22772->13878|22862->13939|22892->13940|22935->13954|23231->14221|23261->14222|23308->14240|23465->14368|23495->14369|23546->14391|23629->14445|23659->14446|23689->14447|23723->14452|23753->14453|23804->14475|23886->14528|23916->14529|23959->14543|23989->14544|24019->14545|24053->14550|24083->14551|24130->14569|24306->14716|24336->14717|24381->14733|24556->14880|24586->14881|24622->14889|24651->14890|24819->15029|24849->15030|24887->15040|24974->15098|25004->15099|25034->15101|25394->15432|25424->15433|25471->15451|25677->15628|25707->15629|25758->15651|26135->15999|26165->16000|26195->16001|26229->16006|26259->16007|26310->16029|26450->16140|26480->16141|26527->16159|26590->16193|26620->16194|26658->16204|26688->16205|26724->16213|26753->16214|26827->16259|26857->16260|26895->16270|26950->16296|26980->16297|27023->16311|27115->16375|27145->16376|27181->16384|27210->16385|27261->16407|27291->16408|27329->16418|27385->16445|27415->16446|27458->16460|27550->16524|27580->16525|27616->16533|27645->16534|27696->16556|27726->16557|27764->16567|27820->16594|27850->16595|27893->16609|27985->16673|28015->16674|28051->16682|28080->16683
                  LINES: 25->1|36->12|36->12|37->13|40->16|40->16|42->18|42->18|42->18|43->19|46->22|46->22|48->24|48->24|48->24|49->25|50->26|50->26|52->28|52->28|52->28|53->29|59->35|59->35|61->37|61->37|61->37|62->38|68->44|68->44|70->46|70->46|70->46|71->47|75->51|75->51|77->53|77->53|77->53|78->54|80->56|80->56|82->58|82->58|82->58|83->59|87->63|87->63|89->65|89->65|89->65|90->66|93->69|93->69|95->71|96->72|96->72|97->73|99->75|99->75|100->76|100->76|100->76|101->77|106->82|106->82|108->84|109->85|109->85|110->86|112->88|112->88|114->90|115->91|115->91|116->92|117->93|117->93|119->95|119->95|119->95|120->96|136->112|136->112|138->114|138->114|138->114|139->115|140->116|140->116|142->118|142->118|142->118|143->119|144->120|144->120|146->122|146->122|146->122|147->123|155->131|155->131|157->133|158->134|158->134|159->135|161->137|161->137|163->139|163->139|164->140|170->146|170->146|172->148|173->149|173->149|174->150|177->153|177->153|178->154|178->154|178->154|179->155|182->158|182->158|183->159|183->159|183->159|184->160|186->162|186->162|188->164|217->193|217->193|218->194|219->195|219->195|220->196|229->205|229->205|230->206|231->207|231->207|232->208|238->214|238->214|241->217|241->217|243->219|244->220|244->220|245->221|246->222|246->222|247->223|254->230|254->230|257->233|257->233|259->235|260->236|260->236|262->238|263->239|263->239|264->240|267->243|267->243|270->246|270->246|271->247|271->247|272->248|272->248|274->250|274->250|275->251|281->257|281->257|282->258|283->259|283->259|285->261|288->264|288->264|289->265|291->267|291->267|292->268|293->269|293->269|294->270|296->272|296->272|298->274|298->274|298->274|299->275|299->275|299->275|300->276|303->279|303->279|304->280|305->281|305->281|306->282|308->284|308->284|309->285|309->285|311->287|313->289|313->289|315->291|315->291|315->291|317->293|342->318|342->318|343->319|358->334|358->334|359->335|361->337|361->337|361->337|361->337|361->337|362->338|369->345|369->345|370->346|370->346|371->347|371->347|373->349|374->350|374->350|375->351|376->352|376->352|377->353|377->353|377->353|378->354|379->355|379->355|380->356|381->357|381->357|381->357|381->357|381->357|382->358|383->359|383->359|384->360|384->360|385->361|385->361|385->361|386->362|387->363|387->363|388->364|389->365|389->365|389->365|389->365|389->365|390->366|391->367|391->367|392->368|392->368|394->370|395->371|395->371|396->372|396->372|396->372|397->373|400->376|400->376|401->377|401->377|405->381|406->382|406->382|407->383|407->383|407->383|408->384|411->387|411->387|412->388|413->389|413->389|414->390|415->391|415->391|415->391|415->391|415->391|416->392|417->393|417->393|418->394|418->394|418->394|418->394|418->394|419->395|422->398|422->398|424->400|429->405|429->405|430->406|430->406|433->409|433->409|434->410|434->410|434->410|435->411|439->415|439->415|440->416|442->418|442->418|443->419|449->425|449->425|449->425|449->425|449->425|450->426|452->428|452->428|453->429|454->430|454->430|455->431|455->431|456->432|456->432|460->436|460->436|461->437|461->437|461->437|462->438|464->440|464->440|465->441|465->441|466->442|466->442|467->443|467->443|467->443|468->444|470->446|470->446|471->447|471->447|472->448|472->448|473->449|473->449|473->449|474->450|476->452|476->452|477->453|477->453
                  -- GENERATED --
              */
          