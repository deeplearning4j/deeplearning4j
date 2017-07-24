
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
//            var currSession;
//            $.get("/arbiter/sessions/current", function(data)"""),format.raw/*207.64*/("""{"""),format.raw/*207.65*/("""
"""),format.raw/*208.1*/("""//                console.log("GOT CURRENT SESSION: " + data);
//                currSession = data;
//            """),format.raw/*210.15*/("""}"""),format.raw/*210.16*/(""")
//                    .fail(function(msg)"""),format.raw/*211.42*/("""{"""),format.raw/*211.43*/("""
"""),format.raw/*212.1*/("""//                        console.log("Failed: " + msg);
//                    """),format.raw/*213.23*/("""}"""),format.raw/*213.24*/(""");

            var currSession;
            $.get("/arbiter/sessions/current", function(data)"""),format.raw/*216.62*/("""{"""),format.raw/*216.63*/("""
                """),format.raw/*217.17*/("""currSession = data; //JSON.stringify(data);
                console.log("Current: " + currSession);
            """),format.raw/*219.13*/("""}"""),format.raw/*219.14*/(""");

            $.get("/arbiter/sessions/all", function(data)"""),format.raw/*221.58*/("""{"""),format.raw/*221.59*/("""
                """),format.raw/*222.17*/("""var keys = data;    // JSON.stringify(data);

                if(keys.length > 1)"""),format.raw/*224.36*/("""{"""),format.raw/*224.37*/("""
                    """),format.raw/*225.21*/("""$("#sessionSelectDiv").show();

                    var elem = $("#sessionSelect");
                    elem.empty();

                    var currSelectedIdx = 0;
                    for (var i = 0; i < keys.length; i++) """),format.raw/*231.59*/("""{"""),format.raw/*231.60*/("""
                        """),format.raw/*232.25*/("""if(keys[i] == currSession)"""),format.raw/*232.51*/("""{"""),format.raw/*232.52*/("""
                            """),format.raw/*233.29*/("""currSelectedIdx = i;
                        """),format.raw/*234.25*/("""}"""),format.raw/*234.26*/("""
                        """),format.raw/*235.25*/("""elem.append("<option value='" + keys[i] + "'>" + keys[i] + "</option>");
                    """),format.raw/*236.21*/("""}"""),format.raw/*236.22*/("""

                    """),format.raw/*238.21*/("""$("#sessionSelect option[value='" + keys[currSelectedIdx] +"']").attr("selected", "selected");
                    $("#sessionSelectDiv").show();
                """),format.raw/*240.17*/("""}"""),format.raw/*240.18*/("""
                """),format.raw/*241.17*/("""console.log("Got sessions: " + keys + ", current: " + currSession);
            """),format.raw/*242.13*/("""}"""),format.raw/*242.14*/(""");


            //Check last update times for each part of document, and update as necessary
            //First section: summary status
            if(lastStatusUpdateTime != statusTime)"""),format.raw/*247.51*/("""{"""),format.raw/*247.52*/("""
                """),format.raw/*248.17*/("""//Get JSON: address set by SummaryStatusResource
                $.get("/arbiter/summary",function(data)"""),format.raw/*249.56*/("""{"""),format.raw/*249.57*/("""
                    """),format.raw/*250.21*/("""var summaryStatusDiv = $('#statusdiv');
                    summaryStatusDiv.html('');

                    var str = JSON.stringify(data);
                    var component = Component.getComponent(str);
                    component.render(summaryStatusDiv);
                """),format.raw/*256.17*/("""}"""),format.raw/*256.18*/(""");

                lastStatusUpdateTime = statusTime;
            """),format.raw/*259.13*/("""}"""),format.raw/*259.14*/("""

            """),format.raw/*261.13*/("""//Second section: Optimization settings
            if(lastSettingsUpdateTime != settingsTime)"""),format.raw/*262.55*/("""{"""),format.raw/*262.56*/("""
                """),format.raw/*263.17*/("""//Get JSON for components
                $.get("/arbiter/config",function(data)"""),format.raw/*264.55*/("""{"""),format.raw/*264.56*/("""
                    """),format.raw/*265.21*/("""var str = JSON.stringify(data);

                    var configDiv = $('#settingsdiv');
                    configDiv.html('');

                    var component = Component.getComponent(str);
                    component.render(configDiv);
                """),format.raw/*272.17*/("""}"""),format.raw/*272.18*/(""");

                lastSettingsUpdateTime = settingsTime;
            """),format.raw/*275.13*/("""}"""),format.raw/*275.14*/("""

            """),format.raw/*277.13*/("""//Third section: Summary results table (summary info for each candidate)
            if(lastResultsUpdateTime != resultsTime)"""),format.raw/*278.53*/("""{"""),format.raw/*278.54*/("""

                """),format.raw/*280.17*/("""//Get JSON; address set by SummaryResultsResource
                $.get("/arbiter/results",function(data)"""),format.raw/*281.56*/("""{"""),format.raw/*281.57*/("""
                    """),format.raw/*282.21*/("""//Expect an array of CandidateInfo type objects here
                    resultsTableContent = data;
                    drawResultTable();
                """),format.raw/*285.17*/("""}"""),format.raw/*285.18*/(""");

                lastResultsUpdateTime = resultsTime;
            """),format.raw/*288.13*/("""}"""),format.raw/*288.14*/("""

            """),format.raw/*290.13*/("""//Finally: Currently selected result
            if(selectedCandidateIdx != null)"""),format.raw/*291.45*/("""{"""),format.raw/*291.46*/("""
                """),format.raw/*292.17*/("""//Get JSON for components
                $.get("/arbiter/candidateInfo/"+selectedCandidateIdx,function(data)"""),format.raw/*293.84*/("""{"""),format.raw/*293.85*/("""
                    """),format.raw/*294.21*/("""var str = JSON.stringify(data);

                    var resultsViewDiv = $('#resultsviewdiv');
                    resultsViewDiv.html('');

                    var component = Component.getComponent(str);
                    component.render(resultsViewDiv);
                """),format.raw/*301.17*/("""}"""),format.raw/*301.18*/(""");
            """),format.raw/*302.13*/("""}"""),format.raw/*302.14*/("""
        """),format.raw/*303.9*/("""}"""),format.raw/*303.10*/(""")
    """),format.raw/*304.5*/("""}"""),format.raw/*304.6*/("""

    """),format.raw/*306.5*/("""function createTable(tableObj,tableId,appendTo)"""),format.raw/*306.52*/("""{"""),format.raw/*306.53*/("""
        """),format.raw/*307.9*/("""//Expect RenderableComponentTable
        var header = tableObj['header'];
        var values = tableObj['table'];
        var title = tableObj['title'];
        var nRows = (values ? values.length : 0);

        if(title)"""),format.raw/*313.18*/("""{"""),format.raw/*313.19*/("""
            """),format.raw/*314.13*/("""appendTo.append("<h5>"+title+"</h5>");
        """),format.raw/*315.9*/("""}"""),format.raw/*315.10*/("""

        """),format.raw/*317.9*/("""var table;
        if(tableId) table = $("<table id=\"" + tableId + "\" class=\"renderableComponentTable\">");
        else table = $("<table class=\"renderableComponentTable\">");
        if(header)"""),format.raw/*320.19*/("""{"""),format.raw/*320.20*/("""
            """),format.raw/*321.13*/("""var headerRow = $("<tr>");
            var len = header.length;
            for( var i=0; i<len; i++ )"""),format.raw/*323.39*/("""{"""),format.raw/*323.40*/("""
                """),format.raw/*324.17*/("""headerRow.append($("<th>" + header[i] + "</th>"));
            """),format.raw/*325.13*/("""}"""),format.raw/*325.14*/("""
            """),format.raw/*326.13*/("""headerRow.append($("</tr>"));
            table.append(headerRow);
        """),format.raw/*328.9*/("""}"""),format.raw/*328.10*/("""

        """),format.raw/*330.9*/("""if(values)"""),format.raw/*330.19*/("""{"""),format.raw/*330.20*/("""
            """),format.raw/*331.13*/("""for( var i=0; i<nRows; i++ )"""),format.raw/*331.41*/("""{"""),format.raw/*331.42*/("""
                """),format.raw/*332.17*/("""var row = $("<tr>");
                var rowValues = values[i];
                var len = rowValues.length;
                for( var j=0; j<len; j++ )"""),format.raw/*335.43*/("""{"""),format.raw/*335.44*/("""
                    """),format.raw/*336.21*/("""row.append($('<td>'+rowValues[j]+'</td>'));
                """),format.raw/*337.17*/("""}"""),format.raw/*337.18*/("""
                """),format.raw/*338.17*/("""row.append($("</tr>"));
                table.append(row);
            """),format.raw/*340.13*/("""}"""),format.raw/*340.14*/("""
        """),format.raw/*341.9*/("""}"""),format.raw/*341.10*/("""

        """),format.raw/*343.9*/("""table.append($("</table>"));
        appendTo.append(table);
    """),format.raw/*345.5*/("""}"""),format.raw/*345.6*/("""

    """),format.raw/*347.5*/("""function drawResultTable()"""),format.raw/*347.31*/("""{"""),format.raw/*347.32*/("""
        """),format.raw/*348.9*/("""//Remove all elements from the table body
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
        for(var i=0; i<len; i++)"""),format.raw/*373.33*/("""{"""),format.raw/*373.34*/("""
            """),format.raw/*374.13*/("""var row;
            if(selectedCandidateIdx == sorted[i][0])"""),format.raw/*375.53*/("""{"""),format.raw/*375.54*/("""
                """),format.raw/*376.17*/("""//Selected row
                row = $('<tr class="resultTableRowSelected" id="rTbl-' + sorted[i][0] + '"/>');
            """),format.raw/*378.13*/("""}"""),format.raw/*378.14*/(""" """),format.raw/*378.15*/("""else """),format.raw/*378.20*/("""{"""),format.raw/*378.21*/("""
                """),format.raw/*379.17*/("""//Normal row
                row = $('<tr class="resultTableRow" id="rTbl-' + sorted[i][0] + '"/>');
            """),format.raw/*381.13*/("""}"""),format.raw/*381.14*/("""
            """),format.raw/*382.13*/("""row.append($("<td>" + sorted[i][0] + "</td>"));
            var score = sorted[i][1];
            row.append($("<td>" + ((!score || score == "null") ? "-" : score) + "</td>"));
            row.append($("<td>" + sorted[i][2] + "</td>"));
            tableBody.append(row);
        """),format.raw/*387.9*/("""}"""),format.raw/*387.10*/("""
    """),format.raw/*388.5*/("""}"""),format.raw/*388.6*/("""

    """),format.raw/*390.5*/("""//Compare function for results, based on sort order
    function compareResultsIndex(a, b)"""),format.raw/*391.39*/("""{"""),format.raw/*391.40*/("""
        """),format.raw/*392.9*/("""return (resultTableSortOrder == "ascending" ? a[0] - b[0] : b[0] - a[0]);
    """),format.raw/*393.5*/("""}"""),format.raw/*393.6*/("""
    """),format.raw/*394.5*/("""function compareScores(a,b)"""),format.raw/*394.32*/("""{"""),format.raw/*394.33*/("""
        """),format.raw/*395.9*/("""//TODO Not always numbers...
        if(resultTableSortOrder == "ascending")"""),format.raw/*396.48*/("""{"""),format.raw/*396.49*/("""
            """),format.raw/*397.13*/("""if(a[1] == "NaN")"""),format.raw/*397.30*/("""{"""),format.raw/*397.31*/("""
                """),format.raw/*398.17*/("""return 1;
            """),format.raw/*399.13*/("""}"""),format.raw/*399.14*/(""" """),format.raw/*399.15*/("""else if(b[1] == "NaN")"""),format.raw/*399.37*/("""{"""),format.raw/*399.38*/("""
                """),format.raw/*400.17*/("""return -1;
            """),format.raw/*401.13*/("""}"""),format.raw/*401.14*/("""
            """),format.raw/*402.13*/("""return a[1] - b[1];
        """),format.raw/*403.9*/("""}"""),format.raw/*403.10*/(""" """),format.raw/*403.11*/("""else """),format.raw/*403.16*/("""{"""),format.raw/*403.17*/("""
            """),format.raw/*404.13*/("""if(a[1] == "NaN")"""),format.raw/*404.30*/("""{"""),format.raw/*404.31*/("""
                """),format.raw/*405.17*/("""return -1;
            """),format.raw/*406.13*/("""}"""),format.raw/*406.14*/(""" """),format.raw/*406.15*/("""else if(b[1] == "NaN")"""),format.raw/*406.37*/("""{"""),format.raw/*406.38*/("""
                """),format.raw/*407.17*/("""return 1;
            """),format.raw/*408.13*/("""}"""),format.raw/*408.14*/("""
            """),format.raw/*409.13*/("""return b[1] - a[1];
        """),format.raw/*410.9*/("""}"""),format.raw/*410.10*/("""
    """),format.raw/*411.5*/("""}"""),format.raw/*411.6*/("""
    """),format.raw/*412.5*/("""function compareStatus(a,b)"""),format.raw/*412.32*/("""{"""),format.raw/*412.33*/("""
        """),format.raw/*413.9*/("""//TODO: secondary sort on... score? index?
        if(resultTableSortOrder == "ascending")"""),format.raw/*414.48*/("""{"""),format.raw/*414.49*/("""
            """),format.raw/*415.13*/("""return (a[2] < b[2] ? -1 : (a[2] > b[2] ? 1 : 0));
        """),format.raw/*416.9*/("""}"""),format.raw/*416.10*/(""" """),format.raw/*416.11*/("""else """),format.raw/*416.16*/("""{"""),format.raw/*416.17*/("""
            """),format.raw/*417.13*/("""return (a[2] < b[2] ? 1 : (a[2] > b[2] ? -1 : 0));
        """),format.raw/*418.9*/("""}"""),format.raw/*418.10*/("""
    """),format.raw/*419.5*/("""}"""),format.raw/*419.6*/("""

    """),format.raw/*421.5*/("""//Do a HTTP request on the specified path, parse and insert into the provided element
    function loadCandidateDetails(path, elementToAppendTo)"""),format.raw/*422.59*/("""{"""),format.raw/*422.60*/("""
        """),format.raw/*423.9*/("""$.get(path, function (data) """),format.raw/*423.37*/("""{"""),format.raw/*423.38*/("""
            """),format.raw/*424.13*/("""var str = JSON.stringify(data);
            var component = Component.getComponent(str);
            component.render(elementToAppendTo);
        """),format.raw/*427.9*/("""}"""),format.raw/*427.10*/(""");
    """),format.raw/*428.5*/("""}"""),format.raw/*428.6*/("""



    """),format.raw/*432.5*/("""//Sorting by column: Intercept click events on table header
    $(function()"""),format.raw/*433.17*/("""{"""),format.raw/*433.18*/("""
        """),format.raw/*434.9*/("""$("#resultsTableHeader").delegate("th", "click", function(e) """),format.raw/*434.70*/("""{"""),format.raw/*434.71*/("""
            """),format.raw/*435.13*/("""//console.log("Header clicked on at: " + $(e.currentTarget).index() + " - " + $(e.currentTarget).html());
            //Update the sort order for the table:
            var clickIndex = $(e.currentTarget).index();
            if(clickIndex == resultTableSortIndex)"""),format.raw/*438.51*/("""{"""),format.raw/*438.52*/("""
                """),format.raw/*439.17*/("""//Switch sort order: ascending -> descending or descending -> ascending
                if(resultTableSortOrder == "ascending")"""),format.raw/*440.56*/("""{"""),format.raw/*440.57*/("""
                    """),format.raw/*441.21*/("""resultTableSortOrder = "descending";
                """),format.raw/*442.17*/("""}"""),format.raw/*442.18*/(""" """),format.raw/*442.19*/("""else """),format.raw/*442.24*/("""{"""),format.raw/*442.25*/("""
                    """),format.raw/*443.21*/("""resultTableSortOrder = "ascending";
                """),format.raw/*444.17*/("""}"""),format.raw/*444.18*/("""
            """),format.raw/*445.13*/("""}"""),format.raw/*445.14*/(""" """),format.raw/*445.15*/("""else """),format.raw/*445.20*/("""{"""),format.raw/*445.21*/("""
                """),format.raw/*446.17*/("""//Sort on column, ascending:
                resultTableSortIndex = clickIndex;
                resultTableSortOrder = "ascending";
            """),format.raw/*449.13*/("""}"""),format.raw/*449.14*/("""

            """),format.raw/*451.13*/("""//Clear record of expanded rows
            expandedRowsCandidateIDs = [];

            //Redraw table
            drawResultTable();
        """),format.raw/*456.9*/("""}"""),format.raw/*456.10*/(""");
    """),format.raw/*457.5*/("""}"""),format.raw/*457.6*/(""");

    //Displaying model/candidate details: Intercept click events on table rows -> toggle selected, fire off update
    $(function()"""),format.raw/*460.17*/("""{"""),format.raw/*460.18*/("""
        """),format.raw/*461.9*/("""$("#resultsTableBody").delegate("tr", "click", function(e)"""),format.raw/*461.67*/("""{"""),format.raw/*461.68*/("""
            """),format.raw/*462.13*/("""var id = this.id;   //Expect: rTbl-X  where X is some index
            var dashIdx = id.indexOf("-");
            var candidateID = Number(id.substring(dashIdx+1));
//            console.log("Clicked row: " + this.id + " with class: " + this.className + ", candidateId = " + candidateID);

            if(this.className == "resultTableRow")"""),format.raw/*467.51*/("""{"""),format.raw/*467.52*/("""
                """),format.raw/*468.17*/("""//Set selected model
                selectedCandidateIdx = candidateID;

                //Fire off update
                doUpdate();
            """),format.raw/*473.13*/("""}"""),format.raw/*473.14*/("""
        """),format.raw/*474.9*/("""}"""),format.raw/*474.10*/(""");
    """),format.raw/*475.5*/("""}"""),format.raw/*475.6*/(""");

    function selectNewSession()"""),format.raw/*477.32*/("""{"""),format.raw/*477.33*/("""
        """),format.raw/*478.9*/("""var selector = $("#sessionSelect");
        var currSelected = selector.val();

        if(currSelected)"""),format.raw/*481.25*/("""{"""),format.raw/*481.26*/("""
            """),format.raw/*482.13*/("""$.ajax("""),format.raw/*482.20*/("""{"""),format.raw/*482.21*/("""
                """),format.raw/*483.17*/("""url: "/arbiter/sessions/set/" + currSelected,
                async: true,
                error: function (query, status, error) """),format.raw/*485.56*/("""{"""),format.raw/*485.57*/("""
                    """),format.raw/*486.21*/("""console.log("Error setting session: " + error);
                """),format.raw/*487.17*/("""}"""),format.raw/*487.18*/(""",
                success: function (data) """),format.raw/*488.42*/("""{"""),format.raw/*488.43*/("""
                    """),format.raw/*489.21*/("""//Update UI immediately
                    doUpdate();
                """),format.raw/*491.17*/("""}"""),format.raw/*491.18*/("""
            """),format.raw/*492.13*/("""}"""),format.raw/*492.14*/(""");
        """),format.raw/*493.9*/("""}"""),format.raw/*493.10*/("""
    """),format.raw/*494.5*/("""}"""),format.raw/*494.6*/("""

"""),format.raw/*496.1*/("""</script>
<script>
    $(function () """),format.raw/*498.19*/("""{"""),format.raw/*498.20*/("""
        """),format.raw/*499.9*/("""$("#accordion").accordion("""),format.raw/*499.35*/("""{"""),format.raw/*499.36*/("""
            """),format.raw/*500.13*/("""collapsible: true,
            heightStyle: "content"
        """),format.raw/*502.9*/("""}"""),format.raw/*502.10*/(""");
    """),format.raw/*503.5*/("""}"""),format.raw/*503.6*/(""");
    $(function () """),format.raw/*504.19*/("""{"""),format.raw/*504.20*/("""
        """),format.raw/*505.9*/("""$("#accordion2").accordion("""),format.raw/*505.36*/("""{"""),format.raw/*505.37*/("""
            """),format.raw/*506.13*/("""collapsible: true,
            heightStyle: "content"
        """),format.raw/*508.9*/("""}"""),format.raw/*508.10*/(""");
    """),format.raw/*509.5*/("""}"""),format.raw/*509.6*/(""");
    $(function () """),format.raw/*510.19*/("""{"""),format.raw/*510.20*/("""
        """),format.raw/*511.9*/("""$("#accordion3").accordion("""),format.raw/*511.36*/("""{"""),format.raw/*511.37*/("""
            """),format.raw/*512.13*/("""collapsible: true,
            heightStyle: "content"
        """),format.raw/*514.9*/("""}"""),format.raw/*514.10*/(""");
    """),format.raw/*515.5*/("""}"""),format.raw/*515.6*/(""");
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
                  DATE: Mon Jul 24 14:04:41 AEST 2017
                  SOURCE: C:/DL4J/Git/Arbiter/arbiter-ui/src/main/views/org/deeplearning4j/arbiter/ui/views/ArbiterUI.scala.html
                  HASH: 6e33dd6e348b6d6f7669ddb0ac5d329050f0c962
                  MATRIX: 647->0|1029->354|1058->355|1100->369|1331->573|1360->574|1399->586|1438->597|1467->598|1509->612|1610->686|1639->687|1678->699|1715->708|1744->709|1786->723|1849->759|1878->760|1917->772|1948->775|1977->776|2019->790|2254->998|2283->999|2322->1011|2353->1014|2382->1015|2424->1029|2661->1239|2690->1240|2729->1252|2763->1258|2792->1259|2834->1273|2938->1350|2967->1351|3006->1363|3053->1382|3082->1383|3124->1397|3296->1542|3325->1543|3364->1555|3460->1623|3489->1624|3531->1638|3673->1753|3702->1754|3741->1766|3791->1788|3820->1789|3862->1803|4033->1947|4062->1948|4101->1960|4151->1982|4180->1983|4222->1997|4352->2100|4381->2101|4420->2113|4580->2245|4609->2246|4651->2260|4804->2386|4833->2387|4870->2397|4927->2426|4956->2427|4998->2441|5254->2670|5283->2671|5322->2683|5408->2741|5437->2742|5479->2756|5572->2822|5601->2823|5640->2835|5777->2944|5806->2945|5848->2959|5928->3011|5958->3012|5998->3024|6046->3043|6076->3044|6119->3058|6702->3613|6732->3614|6772->3626|6819->3644|6849->3645|6892->3659|6951->3690|6981->3691|7021->3703|7074->3727|7104->3728|7147->3742|7206->3773|7236->3774|7276->3786|7406->3886|7437->3887|7480->3901|7918->4311|7948->4312|7988->4324|8050->4357|8080->4358|8123->4372|8454->4675|8484->4676|8524->4688|8586->4721|8616->4722|8659->4736|8770->4819|8800->4820|8838->4830|8890->4853|8920->4854|8963->4868|9081->4958|9111->4959|9149->4969|9189->4980|9219->4981|9262->4995|9355->5060|9385->5061|9425->5073|10606->6225|10636->6226|10674->6236|10805->6338|10835->6339|10878->6353|11446->6892|11476->6893|11506->6895|11652->7012|11682->7013|11755->7057|11785->7058|11815->7060|11924->7140|11954->7141|12080->7238|12110->7239|12157->7257|12300->7371|12330->7372|12422->7435|12452->7436|12499->7454|12611->7537|12641->7538|12692->7560|12949->7788|12979->7789|13034->7815|13089->7841|13119->7842|13178->7872|13253->7918|13283->7919|13338->7945|13461->8039|13491->8040|13544->8064|13737->8228|13767->8229|13814->8247|13924->8328|13954->8329|14176->8522|14206->8523|14253->8541|14387->8646|14417->8647|14468->8669|14780->8952|14810->8953|14909->9023|14939->9024|14984->9040|15108->9135|15138->9136|15185->9154|15295->9235|15325->9236|15376->9258|15671->9524|15701->9525|15804->9599|15834->9600|15879->9616|16034->9742|16064->9743|16113->9763|16248->9869|16278->9870|16329->9892|16517->10051|16547->10052|16648->10124|16678->10125|16723->10141|16834->10223|16864->10224|16911->10242|17050->10352|17080->10353|17131->10375|17444->10659|17474->10660|17519->10676|17549->10677|17587->10687|17617->10688|17652->10695|17681->10696|17717->10704|17793->10751|17823->10752|17861->10762|18118->10990|18148->10991|18191->11005|18267->11053|18297->11054|18337->11066|18568->11268|18598->11269|18641->11283|18774->11387|18804->11388|18851->11406|18944->11470|18974->11471|19017->11485|19122->11562|19152->11563|19192->11575|19231->11585|19261->11586|19304->11600|19361->11628|19391->11629|19438->11647|19620->11800|19650->11801|19701->11823|19791->11884|19821->11885|19868->11903|19970->11976|20000->11977|20038->11987|20068->11988|20108->12000|20203->12067|20232->12068|20268->12076|20323->12102|20353->12103|20391->12113|21748->13441|21778->13442|21821->13456|21912->13518|21942->13519|21989->13537|22143->13662|22173->13663|22203->13664|22237->13669|22267->13670|22314->13688|22458->13803|22488->13804|22531->13818|22844->14103|22874->14104|22908->14110|22937->14111|22973->14119|23093->14210|23123->14211|23161->14221|23268->14300|23297->14301|23331->14307|23387->14334|23417->14335|23455->14345|23561->14422|23591->14423|23634->14437|23680->14454|23710->14455|23757->14473|23809->14496|23839->14497|23869->14498|23920->14520|23950->14521|23997->14539|24050->14563|24080->14564|24123->14578|24180->14607|24210->14608|24240->14609|24274->14614|24304->14615|24347->14629|24393->14646|24423->14647|24470->14665|24523->14689|24553->14690|24583->14691|24634->14713|24664->14714|24711->14732|24763->14755|24793->14756|24836->14770|24893->14799|24923->14800|24957->14806|24986->14807|25020->14813|25076->14840|25106->14841|25144->14851|25264->14942|25294->14943|25337->14957|25425->15017|25455->15018|25485->15019|25519->15024|25549->15025|25592->15039|25680->15099|25710->15100|25744->15106|25773->15107|25809->15115|25983->15260|26013->15261|26051->15271|26108->15299|26138->15300|26181->15314|26358->15463|26388->15464|26424->15472|26453->15473|26493->15485|26599->15562|26629->15563|26667->15573|26757->15634|26787->15635|26830->15649|27126->15916|27156->15917|27203->15935|27360->16063|27390->16064|27441->16086|27524->16140|27554->16141|27584->16142|27618->16147|27648->16148|27699->16170|27781->16223|27811->16224|27854->16238|27884->16239|27914->16240|27948->16245|27978->16246|28025->16264|28201->16411|28231->16412|28276->16428|28451->16575|28481->16576|28517->16584|28546->16585|28713->16723|28743->16724|28781->16734|28868->16792|28898->16793|28941->16807|29316->17153|29346->17154|29393->17172|29575->17325|29605->17326|29643->17336|29673->17337|29709->17345|29738->17346|29804->17383|29834->17384|29872->17394|30008->17501|30038->17502|30081->17516|30117->17523|30147->17524|30194->17542|30355->17674|30385->17675|30436->17697|30530->17762|30560->17763|30633->17807|30663->17808|30714->17830|30817->17904|30847->17905|30890->17919|30920->17920|30960->17932|30990->17933|31024->17939|31053->17940|31085->17944|31153->17983|31183->17984|31221->17994|31276->18020|31306->18021|31349->18035|31441->18099|31471->18100|31507->18108|31536->18109|31587->18131|31617->18132|31655->18142|31711->18169|31741->18170|31784->18184|31876->18248|31906->18249|31942->18257|31971->18258|32022->18280|32052->18281|32090->18291|32146->18318|32176->18319|32219->18333|32311->18397|32341->18398|32377->18406|32406->18407
                  LINES: 25->1|35->11|35->11|36->12|42->18|42->18|44->20|44->20|44->20|45->21|48->24|48->24|50->26|50->26|50->26|51->27|52->28|52->28|54->30|54->30|54->30|55->31|61->37|61->37|63->39|63->39|63->39|64->40|70->46|70->46|72->48|72->48|72->48|73->49|75->51|75->51|77->53|77->53|77->53|78->54|82->58|82->58|84->60|84->60|84->60|85->61|87->63|87->63|89->65|89->65|89->65|90->66|94->70|94->70|96->72|96->72|96->72|97->73|100->76|100->76|102->78|103->79|103->79|104->80|106->82|106->82|107->83|107->83|107->83|108->84|113->89|113->89|115->91|116->92|116->92|117->93|119->95|119->95|121->97|122->98|122->98|123->99|124->100|124->100|126->102|126->102|126->102|127->103|143->119|143->119|145->121|145->121|145->121|146->122|147->123|147->123|149->125|149->125|149->125|150->126|151->127|151->127|153->129|153->129|153->129|154->130|162->138|162->138|164->140|164->140|164->140|165->141|171->147|171->147|173->149|174->150|174->150|175->151|178->154|178->154|179->155|179->155|179->155|180->156|183->159|183->159|184->160|184->160|184->160|185->161|187->163|187->163|189->165|219->195|219->195|220->196|221->197|221->197|222->198|231->207|231->207|232->208|234->210|234->210|235->211|235->211|236->212|237->213|237->213|240->216|240->216|241->217|243->219|243->219|245->221|245->221|246->222|248->224|248->224|249->225|255->231|255->231|256->232|256->232|256->232|257->233|258->234|258->234|259->235|260->236|260->236|262->238|264->240|264->240|265->241|266->242|266->242|271->247|271->247|272->248|273->249|273->249|274->250|280->256|280->256|283->259|283->259|285->261|286->262|286->262|287->263|288->264|288->264|289->265|296->272|296->272|299->275|299->275|301->277|302->278|302->278|304->280|305->281|305->281|306->282|309->285|309->285|312->288|312->288|314->290|315->291|315->291|316->292|317->293|317->293|318->294|325->301|325->301|326->302|326->302|327->303|327->303|328->304|328->304|330->306|330->306|330->306|331->307|337->313|337->313|338->314|339->315|339->315|341->317|344->320|344->320|345->321|347->323|347->323|348->324|349->325|349->325|350->326|352->328|352->328|354->330|354->330|354->330|355->331|355->331|355->331|356->332|359->335|359->335|360->336|361->337|361->337|362->338|364->340|364->340|365->341|365->341|367->343|369->345|369->345|371->347|371->347|371->347|372->348|397->373|397->373|398->374|399->375|399->375|400->376|402->378|402->378|402->378|402->378|402->378|403->379|405->381|405->381|406->382|411->387|411->387|412->388|412->388|414->390|415->391|415->391|416->392|417->393|417->393|418->394|418->394|418->394|419->395|420->396|420->396|421->397|421->397|421->397|422->398|423->399|423->399|423->399|423->399|423->399|424->400|425->401|425->401|426->402|427->403|427->403|427->403|427->403|427->403|428->404|428->404|428->404|429->405|430->406|430->406|430->406|430->406|430->406|431->407|432->408|432->408|433->409|434->410|434->410|435->411|435->411|436->412|436->412|436->412|437->413|438->414|438->414|439->415|440->416|440->416|440->416|440->416|440->416|441->417|442->418|442->418|443->419|443->419|445->421|446->422|446->422|447->423|447->423|447->423|448->424|451->427|451->427|452->428|452->428|456->432|457->433|457->433|458->434|458->434|458->434|459->435|462->438|462->438|463->439|464->440|464->440|465->441|466->442|466->442|466->442|466->442|466->442|467->443|468->444|468->444|469->445|469->445|469->445|469->445|469->445|470->446|473->449|473->449|475->451|480->456|480->456|481->457|481->457|484->460|484->460|485->461|485->461|485->461|486->462|491->467|491->467|492->468|497->473|497->473|498->474|498->474|499->475|499->475|501->477|501->477|502->478|505->481|505->481|506->482|506->482|506->482|507->483|509->485|509->485|510->486|511->487|511->487|512->488|512->488|513->489|515->491|515->491|516->492|516->492|517->493|517->493|518->494|518->494|520->496|522->498|522->498|523->499|523->499|523->499|524->500|526->502|526->502|527->503|527->503|528->504|528->504|529->505|529->505|529->505|530->506|532->508|532->508|533->509|533->509|534->510|534->510|535->511|535->511|535->511|536->512|538->514|538->514|539->515|539->515
                  -- GENERATED --
              */
          