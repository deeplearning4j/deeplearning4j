
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

            //Update available sessions:
            var currSession;
            $.get("/arbiter/sessions/current", function(data)"""),format.raw/*207.62*/("""{"""),format.raw/*207.63*/("""
                """),format.raw/*208.17*/("""currSession = data; //JSON.stringify(data);
                console.log("Current: " + currSession);
            """),format.raw/*210.13*/("""}"""),format.raw/*210.14*/(""");

            $.get("/arbiter/sessions/all", function(data)"""),format.raw/*212.58*/("""{"""),format.raw/*212.59*/("""
                """),format.raw/*213.17*/("""var keys = data;    // JSON.stringify(data);

                if(keys.length > 1)"""),format.raw/*215.36*/("""{"""),format.raw/*215.37*/("""
                    """),format.raw/*216.21*/("""$("#sessionSelectDiv").show();

                    var elem = $("#sessionSelect");
                    elem.empty();

                    var currSelectedIdx = 0;
                    for (var i = 0; i < keys.length; i++) """),format.raw/*222.59*/("""{"""),format.raw/*222.60*/("""
                        """),format.raw/*223.25*/("""if(keys[i] == currSession)"""),format.raw/*223.51*/("""{"""),format.raw/*223.52*/("""
                            """),format.raw/*224.29*/("""currSelectedIdx = i;
                        """),format.raw/*225.25*/("""}"""),format.raw/*225.26*/("""
                        """),format.raw/*226.25*/("""elem.append("<option value='" + keys[i] + "'>" + keys[i] + "</option>");
                    """),format.raw/*227.21*/("""}"""),format.raw/*227.22*/("""

                    """),format.raw/*229.21*/("""$("#sessionSelect option[value='" + keys[currSelectedIdx] +"']").attr("selected", "selected");
                    $("#sessionSelectDiv").show();
                """),format.raw/*231.17*/("""}"""),format.raw/*231.18*/("""
"""),format.raw/*232.1*/("""//                console.log("Got sessions: " + keys + ", current: " + currSession);
            """),format.raw/*233.13*/("""}"""),format.raw/*233.14*/(""");


            //Check last update times for each part of document, and update as necessary
            //First section: summary status
            if(lastStatusUpdateTime != statusTime)"""),format.raw/*238.51*/("""{"""),format.raw/*238.52*/("""
                """),format.raw/*239.17*/("""//Get JSON: address set by SummaryStatusResource
                $.get("/arbiter/summary",function(data)"""),format.raw/*240.56*/("""{"""),format.raw/*240.57*/("""
                    """),format.raw/*241.21*/("""var summaryStatusDiv = $('#statusdiv');
                    summaryStatusDiv.html('');

                    var str = JSON.stringify(data);
                    var component = Component.getComponent(str);
                    component.render(summaryStatusDiv);
                """),format.raw/*247.17*/("""}"""),format.raw/*247.18*/(""");

                lastStatusUpdateTime = statusTime;
            """),format.raw/*250.13*/("""}"""),format.raw/*250.14*/("""

            """),format.raw/*252.13*/("""//Second section: Optimization settings
            if(lastSettingsUpdateTime != settingsTime)"""),format.raw/*253.55*/("""{"""),format.raw/*253.56*/("""
                """),format.raw/*254.17*/("""//Get JSON for components
                $.get("/arbiter/config",function(data)"""),format.raw/*255.55*/("""{"""),format.raw/*255.56*/("""
                    """),format.raw/*256.21*/("""var str = JSON.stringify(data);

                    var configDiv = $('#settingsdiv');
                    configDiv.html('');

                    var component = Component.getComponent(str);
                    component.render(configDiv);
                """),format.raw/*263.17*/("""}"""),format.raw/*263.18*/(""");

                lastSettingsUpdateTime = settingsTime;
            """),format.raw/*266.13*/("""}"""),format.raw/*266.14*/("""

            """),format.raw/*268.13*/("""//Third section: Summary results table (summary info for each candidate)
            if(lastResultsUpdateTime != resultsTime)"""),format.raw/*269.53*/("""{"""),format.raw/*269.54*/("""

                """),format.raw/*271.17*/("""//Get JSON; address set by SummaryResultsResource
                $.get("/arbiter/results",function(data)"""),format.raw/*272.56*/("""{"""),format.raw/*272.57*/("""
                    """),format.raw/*273.21*/("""//Expect an array of CandidateInfo type objects here
                    resultsTableContent = data;
                    drawResultTable();
                """),format.raw/*276.17*/("""}"""),format.raw/*276.18*/(""");

                lastResultsUpdateTime = resultsTime;
            """),format.raw/*279.13*/("""}"""),format.raw/*279.14*/("""

            """),format.raw/*281.13*/("""//Finally: Currently selected result
            if(selectedCandidateIdx != null)"""),format.raw/*282.45*/("""{"""),format.raw/*282.46*/("""
                """),format.raw/*283.17*/("""//Get JSON for components
                $.get("/arbiter/candidateInfo/"+selectedCandidateIdx,function(data)"""),format.raw/*284.84*/("""{"""),format.raw/*284.85*/("""
                    """),format.raw/*285.21*/("""var str = JSON.stringify(data);

                    var resultsViewDiv = $('#resultsviewdiv');
                    resultsViewDiv.html('');

                    var component = Component.getComponent(str);
                    component.render(resultsViewDiv);
                """),format.raw/*292.17*/("""}"""),format.raw/*292.18*/(""");
            """),format.raw/*293.13*/("""}"""),format.raw/*293.14*/("""
        """),format.raw/*294.9*/("""}"""),format.raw/*294.10*/(""")
    """),format.raw/*295.5*/("""}"""),format.raw/*295.6*/("""

    """),format.raw/*297.5*/("""function createTable(tableObj,tableId,appendTo)"""),format.raw/*297.52*/("""{"""),format.raw/*297.53*/("""
        """),format.raw/*298.9*/("""//Expect RenderableComponentTable
        var header = tableObj['header'];
        var values = tableObj['table'];
        var title = tableObj['title'];
        var nRows = (values ? values.length : 0);

        if(title)"""),format.raw/*304.18*/("""{"""),format.raw/*304.19*/("""
            """),format.raw/*305.13*/("""appendTo.append("<h5>"+title+"</h5>");
        """),format.raw/*306.9*/("""}"""),format.raw/*306.10*/("""

        """),format.raw/*308.9*/("""var table;
        if(tableId) table = $("<table id=\"" + tableId + "\" class=\"renderableComponentTable\">");
        else table = $("<table class=\"renderableComponentTable\">");
        if(header)"""),format.raw/*311.19*/("""{"""),format.raw/*311.20*/("""
            """),format.raw/*312.13*/("""var headerRow = $("<tr>");
            var len = header.length;
            for( var i=0; i<len; i++ )"""),format.raw/*314.39*/("""{"""),format.raw/*314.40*/("""
                """),format.raw/*315.17*/("""headerRow.append($("<th>" + header[i] + "</th>"));
            """),format.raw/*316.13*/("""}"""),format.raw/*316.14*/("""
            """),format.raw/*317.13*/("""headerRow.append($("</tr>"));
            table.append(headerRow);
        """),format.raw/*319.9*/("""}"""),format.raw/*319.10*/("""

        """),format.raw/*321.9*/("""if(values)"""),format.raw/*321.19*/("""{"""),format.raw/*321.20*/("""
            """),format.raw/*322.13*/("""for( var i=0; i<nRows; i++ )"""),format.raw/*322.41*/("""{"""),format.raw/*322.42*/("""
                """),format.raw/*323.17*/("""var row = $("<tr>");
                var rowValues = values[i];
                var len = rowValues.length;
                for( var j=0; j<len; j++ )"""),format.raw/*326.43*/("""{"""),format.raw/*326.44*/("""
                    """),format.raw/*327.21*/("""row.append($('<td>'+rowValues[j]+'</td>'));
                """),format.raw/*328.17*/("""}"""),format.raw/*328.18*/("""
                """),format.raw/*329.17*/("""row.append($("</tr>"));
                table.append(row);
            """),format.raw/*331.13*/("""}"""),format.raw/*331.14*/("""
        """),format.raw/*332.9*/("""}"""),format.raw/*332.10*/("""

        """),format.raw/*334.9*/("""table.append($("</table>"));
        appendTo.append(table);
    """),format.raw/*336.5*/("""}"""),format.raw/*336.6*/("""

    """),format.raw/*338.5*/("""function drawResultTable()"""),format.raw/*338.31*/("""{"""),format.raw/*338.32*/("""
        """),format.raw/*339.9*/("""//Remove all elements from the table body
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
        for(var i=0; i<len; i++)"""),format.raw/*364.33*/("""{"""),format.raw/*364.34*/("""
            """),format.raw/*365.13*/("""var row;
            if(selectedCandidateIdx == sorted[i][0])"""),format.raw/*366.53*/("""{"""),format.raw/*366.54*/("""
                """),format.raw/*367.17*/("""//Selected row
                row = $('<tr class="resultTableRowSelected" id="rTbl-' + sorted[i][0] + '"/>');
            """),format.raw/*369.13*/("""}"""),format.raw/*369.14*/(""" """),format.raw/*369.15*/("""else """),format.raw/*369.20*/("""{"""),format.raw/*369.21*/("""
                """),format.raw/*370.17*/("""//Normal row
                row = $('<tr class="resultTableRow" id="rTbl-' + sorted[i][0] + '"/>');
            """),format.raw/*372.13*/("""}"""),format.raw/*372.14*/("""
            """),format.raw/*373.13*/("""row.append($("<td>" + sorted[i][0] + "</td>"));
            var score = sorted[i][1];
            row.append($("<td>" + ((!score || score == "null") ? "-" : score) + "</td>"));
            row.append($("<td>" + sorted[i][2] + "</td>"));
            tableBody.append(row);
        """),format.raw/*378.9*/("""}"""),format.raw/*378.10*/("""
    """),format.raw/*379.5*/("""}"""),format.raw/*379.6*/("""

    """),format.raw/*381.5*/("""//Compare function for results, based on sort order
    function compareResultsIndex(a, b)"""),format.raw/*382.39*/("""{"""),format.raw/*382.40*/("""
        """),format.raw/*383.9*/("""return (resultTableSortOrder == "ascending" ? a[0] - b[0] : b[0] - a[0]);
    """),format.raw/*384.5*/("""}"""),format.raw/*384.6*/("""
    """),format.raw/*385.5*/("""function compareScores(a,b)"""),format.raw/*385.32*/("""{"""),format.raw/*385.33*/("""
        """),format.raw/*386.9*/("""//TODO Not always numbers...
        if(resultTableSortOrder == "ascending")"""),format.raw/*387.48*/("""{"""),format.raw/*387.49*/("""
            """),format.raw/*388.13*/("""if(a[1] == "NaN")"""),format.raw/*388.30*/("""{"""),format.raw/*388.31*/("""
                """),format.raw/*389.17*/("""return 1;
            """),format.raw/*390.13*/("""}"""),format.raw/*390.14*/(""" """),format.raw/*390.15*/("""else if(b[1] == "NaN")"""),format.raw/*390.37*/("""{"""),format.raw/*390.38*/("""
                """),format.raw/*391.17*/("""return -1;
            """),format.raw/*392.13*/("""}"""),format.raw/*392.14*/("""
            """),format.raw/*393.13*/("""return a[1] - b[1];
        """),format.raw/*394.9*/("""}"""),format.raw/*394.10*/(""" """),format.raw/*394.11*/("""else """),format.raw/*394.16*/("""{"""),format.raw/*394.17*/("""
            """),format.raw/*395.13*/("""if(a[1] == "NaN")"""),format.raw/*395.30*/("""{"""),format.raw/*395.31*/("""
                """),format.raw/*396.17*/("""return -1;
            """),format.raw/*397.13*/("""}"""),format.raw/*397.14*/(""" """),format.raw/*397.15*/("""else if(b[1] == "NaN")"""),format.raw/*397.37*/("""{"""),format.raw/*397.38*/("""
                """),format.raw/*398.17*/("""return 1;
            """),format.raw/*399.13*/("""}"""),format.raw/*399.14*/("""
            """),format.raw/*400.13*/("""return b[1] - a[1];
        """),format.raw/*401.9*/("""}"""),format.raw/*401.10*/("""
    """),format.raw/*402.5*/("""}"""),format.raw/*402.6*/("""
    """),format.raw/*403.5*/("""function compareStatus(a,b)"""),format.raw/*403.32*/("""{"""),format.raw/*403.33*/("""
        """),format.raw/*404.9*/("""//TODO: secondary sort on... score? index?
        if(resultTableSortOrder == "ascending")"""),format.raw/*405.48*/("""{"""),format.raw/*405.49*/("""
            """),format.raw/*406.13*/("""return (a[2] < b[2] ? -1 : (a[2] > b[2] ? 1 : 0));
        """),format.raw/*407.9*/("""}"""),format.raw/*407.10*/(""" """),format.raw/*407.11*/("""else """),format.raw/*407.16*/("""{"""),format.raw/*407.17*/("""
            """),format.raw/*408.13*/("""return (a[2] < b[2] ? 1 : (a[2] > b[2] ? -1 : 0));
        """),format.raw/*409.9*/("""}"""),format.raw/*409.10*/("""
    """),format.raw/*410.5*/("""}"""),format.raw/*410.6*/("""

    """),format.raw/*412.5*/("""//Do a HTTP request on the specified path, parse and insert into the provided element
    function loadCandidateDetails(path, elementToAppendTo)"""),format.raw/*413.59*/("""{"""),format.raw/*413.60*/("""
        """),format.raw/*414.9*/("""$.get(path, function (data) """),format.raw/*414.37*/("""{"""),format.raw/*414.38*/("""
            """),format.raw/*415.13*/("""var str = JSON.stringify(data);
            var component = Component.getComponent(str);
            component.render(elementToAppendTo);
        """),format.raw/*418.9*/("""}"""),format.raw/*418.10*/(""");
    """),format.raw/*419.5*/("""}"""),format.raw/*419.6*/("""



    """),format.raw/*423.5*/("""//Sorting by column: Intercept click events on table header
    $(function()"""),format.raw/*424.17*/("""{"""),format.raw/*424.18*/("""
        """),format.raw/*425.9*/("""$("#resultsTableHeader").delegate("th", "click", function(e) """),format.raw/*425.70*/("""{"""),format.raw/*425.71*/("""
            """),format.raw/*426.13*/("""//console.log("Header clicked on at: " + $(e.currentTarget).index() + " - " + $(e.currentTarget).html());
            //Update the sort order for the table:
            var clickIndex = $(e.currentTarget).index();
            if(clickIndex == resultTableSortIndex)"""),format.raw/*429.51*/("""{"""),format.raw/*429.52*/("""
                """),format.raw/*430.17*/("""//Switch sort order: ascending -> descending or descending -> ascending
                if(resultTableSortOrder == "ascending")"""),format.raw/*431.56*/("""{"""),format.raw/*431.57*/("""
                    """),format.raw/*432.21*/("""resultTableSortOrder = "descending";
                """),format.raw/*433.17*/("""}"""),format.raw/*433.18*/(""" """),format.raw/*433.19*/("""else """),format.raw/*433.24*/("""{"""),format.raw/*433.25*/("""
                    """),format.raw/*434.21*/("""resultTableSortOrder = "ascending";
                """),format.raw/*435.17*/("""}"""),format.raw/*435.18*/("""
            """),format.raw/*436.13*/("""}"""),format.raw/*436.14*/(""" """),format.raw/*436.15*/("""else """),format.raw/*436.20*/("""{"""),format.raw/*436.21*/("""
                """),format.raw/*437.17*/("""//Sort on column, ascending:
                resultTableSortIndex = clickIndex;
                resultTableSortOrder = "ascending";
            """),format.raw/*440.13*/("""}"""),format.raw/*440.14*/("""

            """),format.raw/*442.13*/("""//Clear record of expanded rows
            expandedRowsCandidateIDs = [];

            //Redraw table
            drawResultTable();
        """),format.raw/*447.9*/("""}"""),format.raw/*447.10*/(""");
    """),format.raw/*448.5*/("""}"""),format.raw/*448.6*/(""");

    //Displaying model/candidate details: Intercept click events on table rows -> toggle selected, fire off update
    $(function()"""),format.raw/*451.17*/("""{"""),format.raw/*451.18*/("""
        """),format.raw/*452.9*/("""$("#resultsTableBody").delegate("tr", "click", function(e)"""),format.raw/*452.67*/("""{"""),format.raw/*452.68*/("""
            """),format.raw/*453.13*/("""var id = this.id;   //Expect: rTbl-X  where X is some index
            var dashIdx = id.indexOf("-");
            var candidateID = Number(id.substring(dashIdx+1));
//            console.log("Clicked row: " + this.id + " with class: " + this.className + ", candidateId = " + candidateID);

            if(this.className == "resultTableRow")"""),format.raw/*458.51*/("""{"""),format.raw/*458.52*/("""
                """),format.raw/*459.17*/("""//Set selected model
                selectedCandidateIdx = candidateID;

                //Fire off update
                doUpdate();
            """),format.raw/*464.13*/("""}"""),format.raw/*464.14*/("""
        """),format.raw/*465.9*/("""}"""),format.raw/*465.10*/(""");
    """),format.raw/*466.5*/("""}"""),format.raw/*466.6*/(""");

    function selectNewSession()"""),format.raw/*468.32*/("""{"""),format.raw/*468.33*/("""
        """),format.raw/*469.9*/("""var selector = $("#sessionSelect");
        var currSelected = selector.val();

        if(currSelected)"""),format.raw/*472.25*/("""{"""),format.raw/*472.26*/("""
            """),format.raw/*473.13*/("""$.ajax("""),format.raw/*473.20*/("""{"""),format.raw/*473.21*/("""
                """),format.raw/*474.17*/("""url: "/arbiter/sessions/set/" + currSelected,
                async: true,
                error: function (query, status, error) """),format.raw/*476.56*/("""{"""),format.raw/*476.57*/("""
                    """),format.raw/*477.21*/("""console.log("Error setting session: " + error);
                """),format.raw/*478.17*/("""}"""),format.raw/*478.18*/(""",
                success: function (data) """),format.raw/*479.42*/("""{"""),format.raw/*479.43*/("""
                    """),format.raw/*480.21*/("""//Update UI immediately
                    doUpdate();
                """),format.raw/*482.17*/("""}"""),format.raw/*482.18*/("""
            """),format.raw/*483.13*/("""}"""),format.raw/*483.14*/(""");
        """),format.raw/*484.9*/("""}"""),format.raw/*484.10*/("""
    """),format.raw/*485.5*/("""}"""),format.raw/*485.6*/("""

"""),format.raw/*487.1*/("""</script>
<script>
    $(function () """),format.raw/*489.19*/("""{"""),format.raw/*489.20*/("""
        """),format.raw/*490.9*/("""$("#accordion").accordion("""),format.raw/*490.35*/("""{"""),format.raw/*490.36*/("""
            """),format.raw/*491.13*/("""collapsible: true,
            heightStyle: "content"
        """),format.raw/*493.9*/("""}"""),format.raw/*493.10*/(""");
    """),format.raw/*494.5*/("""}"""),format.raw/*494.6*/(""");
    $(function () """),format.raw/*495.19*/("""{"""),format.raw/*495.20*/("""
        """),format.raw/*496.9*/("""$("#accordion2").accordion("""),format.raw/*496.36*/("""{"""),format.raw/*496.37*/("""
            """),format.raw/*497.13*/("""collapsible: true,
            heightStyle: "content"
        """),format.raw/*499.9*/("""}"""),format.raw/*499.10*/(""");
    """),format.raw/*500.5*/("""}"""),format.raw/*500.6*/(""");
    $(function () """),format.raw/*501.19*/("""{"""),format.raw/*501.20*/("""
        """),format.raw/*502.9*/("""$("#accordion3").accordion("""),format.raw/*502.36*/("""{"""),format.raw/*502.37*/("""
            """),format.raw/*503.13*/("""collapsible: true,
            heightStyle: "content"
        """),format.raw/*505.9*/("""}"""),format.raw/*505.10*/(""");
    """),format.raw/*506.5*/("""}"""),format.raw/*506.6*/(""");
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
                  DATE: Mon Jul 24 17:41:55 AEST 2017
                  SOURCE: C:/DL4J/Git/Arbiter/arbiter-ui/src/main/views/org/deeplearning4j/arbiter/ui/views/ArbiterUI.scala.html
                  HASH: 69fc54ccfc8d1508d28120ee9bd89a9652cd6465
                  MATRIX: 647->0|1029->354|1058->355|1100->369|1331->573|1360->574|1399->586|1438->597|1467->598|1509->612|1610->686|1639->687|1678->699|1715->708|1744->709|1786->723|1849->759|1878->760|1917->772|1948->775|1977->776|2019->790|2254->998|2283->999|2322->1011|2353->1014|2382->1015|2424->1029|2661->1239|2690->1240|2729->1252|2763->1258|2792->1259|2834->1273|2938->1350|2967->1351|3006->1363|3053->1382|3082->1383|3124->1397|3296->1542|3325->1543|3364->1555|3460->1623|3489->1624|3531->1638|3673->1753|3702->1754|3741->1766|3791->1788|3820->1789|3862->1803|4033->1947|4062->1948|4101->1960|4151->1982|4180->1983|4222->1997|4352->2100|4381->2101|4420->2113|4580->2245|4609->2246|4651->2260|4804->2386|4833->2387|4870->2397|4927->2426|4956->2427|4998->2441|5254->2670|5283->2671|5322->2683|5408->2741|5437->2742|5479->2756|5572->2822|5601->2823|5640->2835|5777->2944|5806->2945|5848->2959|5928->3011|5958->3012|5998->3024|6046->3043|6076->3044|6119->3058|6702->3613|6732->3614|6772->3626|6819->3644|6849->3645|6892->3659|6951->3690|6981->3691|7021->3703|7074->3727|7104->3728|7147->3742|7206->3773|7236->3774|7276->3786|7406->3886|7437->3887|7480->3901|7918->4311|7948->4312|7988->4324|8050->4357|8080->4358|8123->4372|8454->4675|8484->4676|8524->4688|8586->4721|8616->4722|8659->4736|8770->4819|8800->4820|8838->4830|8890->4853|8920->4854|8963->4868|9081->4958|9111->4959|9149->4969|9189->4980|9219->4981|9262->4995|9355->5060|9385->5061|9425->5073|10606->6225|10636->6226|10674->6236|10805->6338|10835->6339|10878->6353|11442->6888|11472->6889|11519->6907|11662->7021|11692->7022|11784->7085|11814->7086|11861->7104|11973->7187|12003->7188|12054->7210|12311->7438|12341->7439|12396->7465|12451->7491|12481->7492|12540->7522|12615->7568|12645->7569|12700->7595|12823->7689|12853->7690|12906->7714|13099->7878|13129->7879|13159->7881|13287->7980|13317->7981|13539->8174|13569->8175|13616->8193|13750->8298|13780->8299|13831->8321|14143->8604|14173->8605|14272->8675|14302->8676|14347->8692|14471->8787|14501->8788|14548->8806|14658->8887|14688->8888|14739->8910|15034->9176|15064->9177|15167->9251|15197->9252|15242->9268|15397->9394|15427->9395|15476->9415|15611->9521|15641->9522|15692->9544|15880->9703|15910->9704|16011->9776|16041->9777|16086->9793|16197->9875|16227->9876|16274->9894|16413->10004|16443->10005|16494->10027|16807->10311|16837->10312|16882->10328|16912->10329|16950->10339|16980->10340|17015->10347|17044->10348|17080->10356|17156->10403|17186->10404|17224->10414|17481->10642|17511->10643|17554->10657|17630->10705|17660->10706|17700->10718|17931->10920|17961->10921|18004->10935|18137->11039|18167->11040|18214->11058|18307->11122|18337->11123|18380->11137|18485->11214|18515->11215|18555->11227|18594->11237|18624->11238|18667->11252|18724->11280|18754->11281|18801->11299|18983->11452|19013->11453|19064->11475|19154->11536|19184->11537|19231->11555|19333->11628|19363->11629|19401->11639|19431->11640|19471->11652|19566->11719|19595->11720|19631->11728|19686->11754|19716->11755|19754->11765|21111->13093|21141->13094|21184->13108|21275->13170|21305->13171|21352->13189|21506->13314|21536->13315|21566->13316|21600->13321|21630->13322|21677->13340|21821->13455|21851->13456|21894->13470|22207->13755|22237->13756|22271->13762|22300->13763|22336->13771|22456->13862|22486->13863|22524->13873|22631->13952|22660->13953|22694->13959|22750->13986|22780->13987|22818->13997|22924->14074|22954->14075|22997->14089|23043->14106|23073->14107|23120->14125|23172->14148|23202->14149|23232->14150|23283->14172|23313->14173|23360->14191|23413->14215|23443->14216|23486->14230|23543->14259|23573->14260|23603->14261|23637->14266|23667->14267|23710->14281|23756->14298|23786->14299|23833->14317|23886->14341|23916->14342|23946->14343|23997->14365|24027->14366|24074->14384|24126->14407|24156->14408|24199->14422|24256->14451|24286->14452|24320->14458|24349->14459|24383->14465|24439->14492|24469->14493|24507->14503|24627->14594|24657->14595|24700->14609|24788->14669|24818->14670|24848->14671|24882->14676|24912->14677|24955->14691|25043->14751|25073->14752|25107->14758|25136->14759|25172->14767|25346->14912|25376->14913|25414->14923|25471->14951|25501->14952|25544->14966|25721->15115|25751->15116|25787->15124|25816->15125|25856->15137|25962->15214|25992->15215|26030->15225|26120->15286|26150->15287|26193->15301|26489->15568|26519->15569|26566->15587|26723->15715|26753->15716|26804->15738|26887->15792|26917->15793|26947->15794|26981->15799|27011->15800|27062->15822|27144->15875|27174->15876|27217->15890|27247->15891|27277->15892|27311->15897|27341->15898|27388->15916|27564->16063|27594->16064|27639->16080|27814->16227|27844->16228|27880->16236|27909->16237|28076->16375|28106->16376|28144->16386|28231->16444|28261->16445|28304->16459|28679->16805|28709->16806|28756->16824|28938->16977|28968->16978|29006->16988|29036->16989|29072->16997|29101->16998|29167->17035|29197->17036|29235->17046|29371->17153|29401->17154|29444->17168|29480->17175|29510->17176|29557->17194|29718->17326|29748->17327|29799->17349|29893->17414|29923->17415|29996->17459|30026->17460|30077->17482|30180->17556|30210->17557|30253->17571|30283->17572|30323->17584|30353->17585|30387->17591|30416->17592|30448->17596|30516->17635|30546->17636|30584->17646|30639->17672|30669->17673|30712->17687|30804->17751|30834->17752|30870->17760|30899->17761|30950->17783|30980->17784|31018->17794|31074->17821|31104->17822|31147->17836|31239->17900|31269->17901|31305->17909|31334->17910|31385->17932|31415->17933|31453->17943|31509->17970|31539->17971|31582->17985|31674->18049|31704->18050|31740->18058|31769->18059
                  LINES: 25->1|35->11|35->11|36->12|42->18|42->18|44->20|44->20|44->20|45->21|48->24|48->24|50->26|50->26|50->26|51->27|52->28|52->28|54->30|54->30|54->30|55->31|61->37|61->37|63->39|63->39|63->39|64->40|70->46|70->46|72->48|72->48|72->48|73->49|75->51|75->51|77->53|77->53|77->53|78->54|82->58|82->58|84->60|84->60|84->60|85->61|87->63|87->63|89->65|89->65|89->65|90->66|94->70|94->70|96->72|96->72|96->72|97->73|100->76|100->76|102->78|103->79|103->79|104->80|106->82|106->82|107->83|107->83|107->83|108->84|113->89|113->89|115->91|116->92|116->92|117->93|119->95|119->95|121->97|122->98|122->98|123->99|124->100|124->100|126->102|126->102|126->102|127->103|143->119|143->119|145->121|145->121|145->121|146->122|147->123|147->123|149->125|149->125|149->125|150->126|151->127|151->127|153->129|153->129|153->129|154->130|162->138|162->138|164->140|164->140|164->140|165->141|171->147|171->147|173->149|174->150|174->150|175->151|178->154|178->154|179->155|179->155|179->155|180->156|183->159|183->159|184->160|184->160|184->160|185->161|187->163|187->163|189->165|219->195|219->195|220->196|221->197|221->197|222->198|231->207|231->207|232->208|234->210|234->210|236->212|236->212|237->213|239->215|239->215|240->216|246->222|246->222|247->223|247->223|247->223|248->224|249->225|249->225|250->226|251->227|251->227|253->229|255->231|255->231|256->232|257->233|257->233|262->238|262->238|263->239|264->240|264->240|265->241|271->247|271->247|274->250|274->250|276->252|277->253|277->253|278->254|279->255|279->255|280->256|287->263|287->263|290->266|290->266|292->268|293->269|293->269|295->271|296->272|296->272|297->273|300->276|300->276|303->279|303->279|305->281|306->282|306->282|307->283|308->284|308->284|309->285|316->292|316->292|317->293|317->293|318->294|318->294|319->295|319->295|321->297|321->297|321->297|322->298|328->304|328->304|329->305|330->306|330->306|332->308|335->311|335->311|336->312|338->314|338->314|339->315|340->316|340->316|341->317|343->319|343->319|345->321|345->321|345->321|346->322|346->322|346->322|347->323|350->326|350->326|351->327|352->328|352->328|353->329|355->331|355->331|356->332|356->332|358->334|360->336|360->336|362->338|362->338|362->338|363->339|388->364|388->364|389->365|390->366|390->366|391->367|393->369|393->369|393->369|393->369|393->369|394->370|396->372|396->372|397->373|402->378|402->378|403->379|403->379|405->381|406->382|406->382|407->383|408->384|408->384|409->385|409->385|409->385|410->386|411->387|411->387|412->388|412->388|412->388|413->389|414->390|414->390|414->390|414->390|414->390|415->391|416->392|416->392|417->393|418->394|418->394|418->394|418->394|418->394|419->395|419->395|419->395|420->396|421->397|421->397|421->397|421->397|421->397|422->398|423->399|423->399|424->400|425->401|425->401|426->402|426->402|427->403|427->403|427->403|428->404|429->405|429->405|430->406|431->407|431->407|431->407|431->407|431->407|432->408|433->409|433->409|434->410|434->410|436->412|437->413|437->413|438->414|438->414|438->414|439->415|442->418|442->418|443->419|443->419|447->423|448->424|448->424|449->425|449->425|449->425|450->426|453->429|453->429|454->430|455->431|455->431|456->432|457->433|457->433|457->433|457->433|457->433|458->434|459->435|459->435|460->436|460->436|460->436|460->436|460->436|461->437|464->440|464->440|466->442|471->447|471->447|472->448|472->448|475->451|475->451|476->452|476->452|476->452|477->453|482->458|482->458|483->459|488->464|488->464|489->465|489->465|490->466|490->466|492->468|492->468|493->469|496->472|496->472|497->473|497->473|497->473|498->474|500->476|500->476|501->477|502->478|502->478|503->479|503->479|504->480|506->482|506->482|507->483|507->483|508->484|508->484|509->485|509->485|511->487|513->489|513->489|514->490|514->490|514->490|515->491|517->493|517->493|518->494|518->494|519->495|519->495|520->496|520->496|520->496|521->497|523->499|523->499|524->500|524->500|525->501|525->501|526->502|526->502|526->502|527->503|529->505|529->505|530->506|530->506
                  -- GENERATED --
              */
          