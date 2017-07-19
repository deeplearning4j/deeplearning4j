
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

        """),format.raw/*122.9*/("""#accordion .ui-accordion-header, #accordion2 .ui-accordion-header """),format.raw/*122.75*/("""{"""),format.raw/*122.76*/("""
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
        $.get("/lastUpdate",function(data)"""),format.raw/*195.43*/("""{"""),format.raw/*195.44*/("""
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
                $.get("/summary",function(data)"""),format.raw/*207.48*/("""{"""),format.raw/*207.49*/("""
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
                $.get("/config",function(data)"""),format.raw/*222.47*/("""{"""),format.raw/*222.48*/("""
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
                $.get("/results",function(data)"""),format.raw/*239.48*/("""{"""),format.raw/*239.49*/("""
                    """),format.raw/*240.21*/("""//Expect an array of CandidateStatus type objects here
                    resultsTableContent = data;
                    drawResultTable();
                """),format.raw/*243.17*/("""}"""),format.raw/*243.18*/(""");

                lastResultsUpdateTime = resultsTime;
            """),format.raw/*246.13*/("""}"""),format.raw/*246.14*/("""
        """),format.raw/*247.9*/("""}"""),format.raw/*247.10*/(""")
    """),format.raw/*248.5*/("""}"""),format.raw/*248.6*/(""",2000);    //Loop every 2 seconds

    function createAndAddComponent(renderableComponent, appendTo)"""),format.raw/*250.66*/("""{"""),format.raw/*250.67*/("""
        """),format.raw/*251.9*/("""var key = Object.keys(renderableComponent)[0];
        var type = renderableComponent[key]['componentType'];

        switch(type)"""),format.raw/*254.21*/("""{"""),format.raw/*254.22*/("""
            case "string":
                var s = renderableComponent[key]['string'];
                appendTo.append(s);
                break;
            case "simpletable":
                createTable(renderableComponent[key],null,appendTo);
                break;
            case "linechart":
                createLineChart(renderableComponent[key],appendTo);
                break;
            case "scatterplot":
                createScatterPlot(renderableComponent[key],appendTo);
                break;
            case "accordion":
                createAccordion(renderableComponent[key],appendTo);
                break;
            default:
                return "(Error rendering component: Unknown object)";
        """),format.raw/*273.9*/("""}"""),format.raw/*273.10*/("""
    """),format.raw/*274.5*/("""}"""),format.raw/*274.6*/("""

    """),format.raw/*276.5*/("""function createTable(tableObj,tableId,appendTo)"""),format.raw/*276.52*/("""{"""),format.raw/*276.53*/("""
        """),format.raw/*277.9*/("""//Expect RenderableComponentTable
        var header = tableObj['header'];
        var values = tableObj['table'];
        var title = tableObj['title'];
        var nRows = (values ? values.length : 0);

        if(title)"""),format.raw/*283.18*/("""{"""),format.raw/*283.19*/("""
            """),format.raw/*284.13*/("""appendTo.append("<h5>"+title+"</h5>");
        """),format.raw/*285.9*/("""}"""),format.raw/*285.10*/("""

        """),format.raw/*287.9*/("""var table;
        if(tableId) table = $("<table id=\"" + tableId + "\" class=\"renderableComponentTable\">");
        else table = $("<table class=\"renderableComponentTable\">");
        if(header)"""),format.raw/*290.19*/("""{"""),format.raw/*290.20*/("""
            """),format.raw/*291.13*/("""var headerRow = $("<tr>");
            var len = header.length;
            for( var i=0; i<len; i++ )"""),format.raw/*293.39*/("""{"""),format.raw/*293.40*/("""
                """),format.raw/*294.17*/("""headerRow.append($("<th>" + header[i] + "</th>"));
            """),format.raw/*295.13*/("""}"""),format.raw/*295.14*/("""
            """),format.raw/*296.13*/("""headerRow.append($("</tr>"));
            table.append(headerRow);
        """),format.raw/*298.9*/("""}"""),format.raw/*298.10*/("""

        """),format.raw/*300.9*/("""if(values)"""),format.raw/*300.19*/("""{"""),format.raw/*300.20*/("""
            """),format.raw/*301.13*/("""for( var i=0; i<nRows; i++ )"""),format.raw/*301.41*/("""{"""),format.raw/*301.42*/("""
                """),format.raw/*302.17*/("""var row = $("<tr>");
                var rowValues = values[i];
                var len = rowValues.length;
                for( var j=0; j<len; j++ )"""),format.raw/*305.43*/("""{"""),format.raw/*305.44*/("""
                    """),format.raw/*306.21*/("""row.append($('<td>'+rowValues[j]+'</td>'));
                """),format.raw/*307.17*/("""}"""),format.raw/*307.18*/("""
                """),format.raw/*308.17*/("""row.append($("</tr>"));
                table.append(row);
            """),format.raw/*310.13*/("""}"""),format.raw/*310.14*/("""
        """),format.raw/*311.9*/("""}"""),format.raw/*311.10*/("""

        """),format.raw/*313.9*/("""table.append($("</table>"));
        appendTo.append(table);
    """),format.raw/*315.5*/("""}"""),format.raw/*315.6*/("""

    """),format.raw/*317.5*/("""/** Create + add line chart with multiple lines, (optional) title, (optional) series names.
     * appendTo: jquery selector of object to append to. MUST HAVE ID
     * */
    function createLineChart(chartObj, appendTo)"""),format.raw/*320.49*/("""{"""),format.raw/*320.50*/("""
        """),format.raw/*321.9*/("""//Expect: RenderableComponentLineChart
        var title = chartObj['title'];
        var xData = chartObj['x'];
        var yData = chartObj['y'];
        var seriesNames = chartObj['seriesNames'];
        var nSeries = (!xData ? 0 : xData.length);
        var title = chartObj['title'];

        // Set the dimensions of the canvas / graph
        var margin = """),format.raw/*330.22*/("""{"""),format.raw/*330.23*/("""top: 60, right: 20, bottom: 60, left: 50"""),format.raw/*330.63*/("""}"""),format.raw/*330.64*/(""",
                width = 650 - margin.left - margin.right,
                height = 350 - margin.top - margin.bottom;

        // Set the ranges
        var xScale = d3.scale.linear().range([0, width]);
        var yScale = d3.scale.linear().range([height, 0]);

        // Define the axes
        var xAxis = d3.svg.axis().scale(xScale)
                .innerTickSize(-height)     //used as grid line
                .orient("bottom").ticks(5);

        var yAxis = d3.svg.axis().scale(yScale)
                .innerTickSize(-width)      //used as grid line
                .orient("left").ticks(5);

        // Define the line
        var valueline = d3.svg.line()
                .x(function(d) """),format.raw/*349.32*/("""{"""),format.raw/*349.33*/(""" """),format.raw/*349.34*/("""return xScale(d.xPos); """),format.raw/*349.57*/("""}"""),format.raw/*349.58*/(""")
                .y(function(d) """),format.raw/*350.32*/("""{"""),format.raw/*350.33*/(""" """),format.raw/*350.34*/("""return yScale(d.yPos); """),format.raw/*350.57*/("""}"""),format.raw/*350.58*/(""");

        // Adds the svg canvas
        var svg = d3.select("#" + appendTo.attr("id"))
                .append("svg")
                .attr("width", width + margin.left + margin.right)
                .attr("height", height + margin.top + margin.bottom)
                .attr("padding", "20px")
                .append("g")
                .attr("transform",
                        "translate(" + margin.left + "," + margin.top + ")");

        // Scale the range of the chart
        var xMax = -Number.MAX_VALUE;
        var yMax = -Number.MAX_VALUE;
        var yMin = Number.MAX_VALUE;
        for( var i=0; i<nSeries; i++)"""),format.raw/*366.38*/("""{"""),format.raw/*366.39*/("""
            """),format.raw/*367.13*/("""var xV = xData[i];
            var yV = yData[i];
            var thisMax = d3.max(xV);
            var thisMaxY = d3.max(yV);
            var thisMinY = d3.min(yV);
            if(thisMax > xMax) xMax = thisMax;
            if(thisMaxY > yMax) yMax = thisMaxY;
            if(thisMinY < yMin) yMin = thisMinY;
        """),format.raw/*375.9*/("""}"""),format.raw/*375.10*/("""
        """),format.raw/*376.9*/("""if(yMin > 0) yMin = 0;
        xScale.domain([0, xMax]);
        yScale.domain([yMin, yMax]);

        // Add the valueline path.
        var color = d3.scale.category10();
        for( var i=0; i<nSeries; i++)"""),format.raw/*382.38*/("""{"""),format.raw/*382.39*/("""
            """),format.raw/*383.13*/("""var xVals = xData[i];
            var yVals = yData[i];

            var data = xVals.map(function(d, i)"""),format.raw/*386.48*/("""{"""),format.raw/*386.49*/("""
                """),format.raw/*387.17*/("""return """),format.raw/*387.24*/("""{"""),format.raw/*387.25*/(""" """),format.raw/*387.26*/("""'xPos' : xVals[i], 'yPos' : yVals[i] """),format.raw/*387.63*/("""}"""),format.raw/*387.64*/(""";
            """),format.raw/*388.13*/("""}"""),format.raw/*388.14*/(""");
            svg.append("path")
                    .attr("class", "line")
                    .style("stroke", color(i))
                    .attr("d", valueline(data));
        """),format.raw/*393.9*/("""}"""),format.raw/*393.10*/("""

        """),format.raw/*395.9*/("""// Add the X Axis
        svg.append("g")
                .attr("class", "x axis")
                .attr("transform", "translate(0," + height + ")")
                .call(xAxis);

        // Add the Y Axis
        svg.append("g")
                .attr("class", "y axis")
                .call(yAxis);

        //Add legend (if present)
        if(seriesNames) """),format.raw/*407.25*/("""{"""),format.raw/*407.26*/("""
            """),format.raw/*408.13*/("""var legendSpace = width / i;
            for (var i = 0; i < nSeries; i++) """),format.raw/*409.47*/("""{"""),format.raw/*409.48*/("""
                """),format.raw/*410.17*/("""var values = xData[i];
                var yValues = yData[i];
                var lastX = values[values.length - 1];
                var lastY = yValues[yValues.length - 1];
                var toDisplay;
                if(!lastX || !lastY) toDisplay = seriesNames[i] + " (no data)";
                else toDisplay = seriesNames[i] + " (" + lastX.toPrecision(5) + "," + lastY.toPrecision(5) + ")";
                svg.append("text")
                        .attr("x", (legendSpace / 2) + i * legendSpace) // spacing
                        .attr("y", height + (margin.bottom / 2) + 5)
                        .attr("class", "legend")    // style the legend
                        .style("fill", color(i))
                        .text(toDisplay);

            """),format.raw/*424.13*/("""}"""),format.raw/*424.14*/("""
        """),format.raw/*425.9*/("""}"""),format.raw/*425.10*/("""

        """),format.raw/*427.9*/("""//Add title (if present)
        if(title)"""),format.raw/*428.18*/("""{"""),format.raw/*428.19*/("""
            """),format.raw/*429.13*/("""svg.append("text")
                    .attr("x", (width / 2))
                    .attr("y", 0 - ((margin.top-30) / 2))
                    .attr("text-anchor", "middle")
                    .style("font-size", "13px")
                    .style("text-decoration", "underline")
                    .text(title);
        """),format.raw/*436.9*/("""}"""),format.raw/*436.10*/("""
    """),format.raw/*437.5*/("""}"""),format.raw/*437.6*/("""

    """),format.raw/*439.5*/("""/** Create + add scatter plot chart with multiple different types of points, (optional) title, (optional) series names.
     * appendTo: jquery selector of object to append to. MUST HAVE ID
     * */
    function createScatterPlot(chartObj, appendTo)"""),format.raw/*442.51*/("""{"""),format.raw/*442.52*/("""
        """),format.raw/*443.9*/("""//TODO modify this to do scatter plot, not line chart
        //Expect: RenderableComponentLineChart
        var title = chartObj['title'];
        var xData = chartObj['x'];
        var yData = chartObj['y'];
        var seriesNames = chartObj['seriesNames'];
        var nSeries = (!xData ? 0 : xData.length);
        var title = chartObj['title'];

        // Set the dimensions of the canvas / graph
        var margin = """),format.raw/*453.22*/("""{"""),format.raw/*453.23*/("""top: 60, right: 20, bottom: 60, left: 50"""),format.raw/*453.63*/("""}"""),format.raw/*453.64*/(""",
                width = 650 - margin.left - margin.right,
                height = 350 - margin.top - margin.bottom;

        // Set the ranges
        var xScale = d3.scale.linear().range([0, width]);
        var yScale = d3.scale.linear().range([height, 0]);

        // Define the axes
        var xAxis = d3.svg.axis().scale(xScale)
                .innerTickSize(-height)     //used as grid line
                .orient("bottom").ticks(5);

        var yAxis = d3.svg.axis().scale(yScale)
                .innerTickSize(-width)      //used as grid line
                .orient("left").ticks(5);

        // Define the line
        var valueline = d3.svg.line()
                .x(function(d) """),format.raw/*472.32*/("""{"""),format.raw/*472.33*/(""" """),format.raw/*472.34*/("""return xScale(d.xPos); """),format.raw/*472.57*/("""}"""),format.raw/*472.58*/(""")
                .y(function(d) """),format.raw/*473.32*/("""{"""),format.raw/*473.33*/(""" """),format.raw/*473.34*/("""return yScale(d.yPos); """),format.raw/*473.57*/("""}"""),format.raw/*473.58*/(""");

        // Adds the svg canvas
        var svg = d3.select("#" + appendTo.attr("id"))
                .append("svg")
                .attr("width", width + margin.left + margin.right)
                .attr("height", height + margin.top + margin.bottom)
                .attr("padding", "20px")
                .append("g")
                .attr("transform",
                        "translate(" + margin.left + "," + margin.top + ")");

        // Scale the range of the chart
        var xMax = -Number.MAX_VALUE;
        var yMax = -Number.MAX_VALUE;
        var yMin = Number.MAX_VALUE;
        for( var i=0; i<nSeries; i++)"""),format.raw/*489.38*/("""{"""),format.raw/*489.39*/("""
            """),format.raw/*490.13*/("""var xV = xData[i];
            var yV = yData[i];
            var thisMax = d3.max(xV);
            var thisMaxY = d3.max(yV);
            var thisMinY = d3.min(yV);
            if(thisMax > xMax) xMax = thisMax;
            if(thisMaxY > yMax) yMax = thisMaxY;
            if(thisMinY < yMin) yMin = thisMinY;
        """),format.raw/*498.9*/("""}"""),format.raw/*498.10*/("""
        """),format.raw/*499.9*/("""if(yMin > 0) yMin = 0;
        xScale.domain([0, xMax]);
        yScale.domain([yMin, yMax]);

        // Add the valueline path.
        var color = d3.scale.category10();
        for( var i=0; i<nSeries; i++)"""),format.raw/*505.38*/("""{"""),format.raw/*505.39*/("""
            """),format.raw/*506.13*/("""var xVals = xData[i];
            var yVals = yData[i];

            var data = xVals.map(function(d, i)"""),format.raw/*509.48*/("""{"""),format.raw/*509.49*/("""
                """),format.raw/*510.17*/("""return """),format.raw/*510.24*/("""{"""),format.raw/*510.25*/(""" """),format.raw/*510.26*/("""'xPos' : xVals[i], 'yPos' : yVals[i] """),format.raw/*510.63*/("""}"""),format.raw/*510.64*/(""";
            """),format.raw/*511.13*/("""}"""),format.raw/*511.14*/(""");

            svg.selectAll("circle")
                    .data(data)
                    .enter()
                    .append("circle")
                    .style("fill", function(d)"""),format.raw/*517.47*/("""{"""),format.raw/*517.48*/(""" """),format.raw/*517.49*/("""return color(i)"""),format.raw/*517.64*/("""}"""),format.raw/*517.65*/(""")
                    .attr("r",3.0)
                    .attr("cx", function(d)"""),format.raw/*519.44*/("""{"""),format.raw/*519.45*/(""" """),format.raw/*519.46*/("""return xScale(d['xPos']); """),format.raw/*519.72*/("""}"""),format.raw/*519.73*/(""")
                    .attr("cy", function(d)"""),format.raw/*520.44*/("""{"""),format.raw/*520.45*/(""" """),format.raw/*520.46*/("""return yScale(d['yPos']); """),format.raw/*520.72*/("""}"""),format.raw/*520.73*/(""");
        """),format.raw/*521.9*/("""}"""),format.raw/*521.10*/("""

        """),format.raw/*523.9*/("""// Add the X Axis
        svg.append("g")
                .attr("class", "x axis")
                .attr("transform", "translate(0," + height + ")")
                .call(xAxis);

        // Add the Y Axis
        svg.append("g")
                .attr("class", "y axis")
                .call(yAxis);

        //Add legend (if present)
        if(seriesNames) """),format.raw/*535.25*/("""{"""),format.raw/*535.26*/("""
            """),format.raw/*536.13*/("""var legendSpace = width / i;
            for (var i = 0; i < nSeries; i++) """),format.raw/*537.47*/("""{"""),format.raw/*537.48*/("""
                """),format.raw/*538.17*/("""var values = xData[i];
                var yValues = yData[i];
                var lastX = values[values.length - 1];
                var lastY = yValues[yValues.length - 1];
                var toDisplay;
                if(!lastX || !lastY) toDisplay = seriesNames[i] + " (no data)";
                else toDisplay = seriesNames[i] + " (" + lastX.toPrecision(5) + "," + lastY.toPrecision(5) + ")";
                svg.append("text")
                        .attr("x", (legendSpace / 2) + i * legendSpace) // spacing
                        .attr("y", height + (margin.bottom / 2) + 5)
                        .attr("class", "legend")    // style the legend
                        .style("fill", color(i))
                        .text(toDisplay);

            """),format.raw/*552.13*/("""}"""),format.raw/*552.14*/("""
        """),format.raw/*553.9*/("""}"""),format.raw/*553.10*/("""

        """),format.raw/*555.9*/("""//Add title (if present)
        if(title)"""),format.raw/*556.18*/("""{"""),format.raw/*556.19*/("""
            """),format.raw/*557.13*/("""svg.append("text")
                    .attr("x", (width / 2))
                    .attr("y", 0 - ((margin.top-30) / 2))
                    .attr("text-anchor", "middle")
                    .style("font-size", "13px")
                    .style("text-decoration", "underline")
                    .text(title);
        """),format.raw/*564.9*/("""}"""),format.raw/*564.10*/("""
    """),format.raw/*565.5*/("""}"""),format.raw/*565.6*/("""

    """),format.raw/*567.5*/("""function createAccordion(accordionObj, appendTo) """),format.raw/*567.54*/("""{"""),format.raw/*567.55*/("""
        """),format.raw/*568.9*/("""var title = accordionObj['title'];
        var defaultCollapsed = accordionObj['defaultCollapsed'];

        var tempDivOuter = $('<div><h3>' + title + '</h3></div>');
        tempDivOuter.uniqueId();
        var generatedIDOuter = tempDivOuter.attr('id');
        var tempDivInner = $('<div></div>');
        tempDivInner.uniqueId();
        var generatedIDInner = tempDivInner.attr('id');
        tempDivOuter.append(tempDivInner);
        appendTo.append(tempDivOuter);

        if (defaultCollapsed == true) """),format.raw/*580.39*/("""{"""),format.raw/*580.40*/("""
            """),format.raw/*581.13*/("""$("#" + generatedIDOuter).accordion("""),format.raw/*581.49*/("""{"""),format.raw/*581.50*/("""collapsible: true, heightStyle: "content", active: false"""),format.raw/*581.106*/("""}"""),format.raw/*581.107*/(""");
        """),format.raw/*582.9*/("""}"""),format.raw/*582.10*/(""" """),format.raw/*582.11*/("""else """),format.raw/*582.16*/("""{"""),format.raw/*582.17*/("""
            """),format.raw/*583.13*/("""$("#" + generatedIDOuter).accordion("""),format.raw/*583.49*/("""{"""),format.raw/*583.50*/("""collapsible: true, heightStyle: "content""""),format.raw/*583.91*/("""}"""),format.raw/*583.92*/(""");
        """),format.raw/*584.9*/("""}"""),format.raw/*584.10*/("""

        """),format.raw/*586.9*/("""//Add the inner components:
        var innerComponents = accordionObj['innerComponents'];
        var len = (!innerComponents ? 0 : innerComponents.length);
        for( var i=0; i<len; i++ )"""),format.raw/*589.35*/("""{"""),format.raw/*589.36*/("""
            """),format.raw/*590.13*/("""var component = innerComponents[i];
            createAndAddComponent(component,$("#"+generatedIDInner));
        """),format.raw/*592.9*/("""}"""),format.raw/*592.10*/("""
    """),format.raw/*593.5*/("""}"""),format.raw/*593.6*/("""

    """),format.raw/*595.5*/("""function drawResultTable()"""),format.raw/*595.31*/("""{"""),format.raw/*595.32*/("""

        """),format.raw/*597.9*/("""//Remove all elements from the table body
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
        for(var i=0; i<len; i++)"""),format.raw/*622.33*/("""{"""),format.raw/*622.34*/("""
            """),format.raw/*623.13*/("""var row = $('<tr class="resultTableRow" id="resultTableRow-' + sorted[i].index + '"/>');
            row.append($("<td>" + sorted[i].index + "</td>"));
            var score = sorted[i].score;
            row.append($("<td>" + ((!score || score == "null") ? "-" : score) + "</td>"));
            row.append($("<td>" + sorted[i].status + "</td>"));
            tableBody.append(row);

            //Create hidden row for expanding:
            var rowID = 'resultTableRow-' + sorted[i].index + '-content';
            var contentRow = $('<tr id="' + rowID + '" class="resultTableRowContent"/>');
            var td3 = $("<td colspan=3 id=" + rowID + "-td></td>");
            td3.append("(Result status - loading)");
            contentRow.append(td3);

            tableBody.append(contentRow);
            if(expandedRowsCandidateIDs.indexOf(sorted[i].index) == -1 )"""),format.raw/*638.73*/("""{"""),format.raw/*638.74*/("""
                """),format.raw/*639.17*/("""contentRow.hide();

            """),format.raw/*641.13*/("""}"""),format.raw/*641.14*/(""" """),format.raw/*641.15*/("""else """),format.raw/*641.20*/("""{"""),format.raw/*641.21*/("""
                """),format.raw/*642.17*/("""//Load info. TODO: make this more efficient (stored info, check for updates, etc)
                td3.empty();

                var path = "/modelResults/" + sorted[i].index;
                loadCandidateDetails(path, td3);

                contentRow.show();
            """),format.raw/*649.13*/("""}"""),format.raw/*649.14*/("""
        """),format.raw/*650.9*/("""}"""),format.raw/*650.10*/("""
    """),format.raw/*651.5*/("""}"""),format.raw/*651.6*/("""

    """),format.raw/*653.5*/("""//Compare function for results, based on sort order
    function compareResultsIndex(a, b)"""),format.raw/*654.39*/("""{"""),format.raw/*654.40*/("""
        """),format.raw/*655.9*/("""return (resultTableSortOrder == "ascending" ? a.index - b.index : b.index - a.index);
    """),format.raw/*656.5*/("""}"""),format.raw/*656.6*/("""
    """),format.raw/*657.5*/("""function compareScores(a,b)"""),format.raw/*657.32*/("""{"""),format.raw/*657.33*/("""
        """),format.raw/*658.9*/("""//TODO Not always numbers...
        if(resultTableSortOrder == "ascending")"""),format.raw/*659.48*/("""{"""),format.raw/*659.49*/("""
            """),format.raw/*660.13*/("""return a.score - b.score;
        """),format.raw/*661.9*/("""}"""),format.raw/*661.10*/(""" """),format.raw/*661.11*/("""else """),format.raw/*661.16*/("""{"""),format.raw/*661.17*/("""
            """),format.raw/*662.13*/("""return b.score - a.score;
        """),format.raw/*663.9*/("""}"""),format.raw/*663.10*/("""
    """),format.raw/*664.5*/("""}"""),format.raw/*664.6*/("""
    """),format.raw/*665.5*/("""function compareStatus(a,b)"""),format.raw/*665.32*/("""{"""),format.raw/*665.33*/("""
        """),format.raw/*666.9*/("""//TODO: secondary sort on... score? index?
        if(resultTableSortOrder == "ascending")"""),format.raw/*667.48*/("""{"""),format.raw/*667.49*/("""
            """),format.raw/*668.13*/("""return (a.status < b.status ? -1 : (a.status > b.status ? 1 : 0));
        """),format.raw/*669.9*/("""}"""),format.raw/*669.10*/(""" """),format.raw/*669.11*/("""else """),format.raw/*669.16*/("""{"""),format.raw/*669.17*/("""
            """),format.raw/*670.13*/("""return (a.status < b.status ? 1 : (a.status > b.status ? -1 : 0));
        """),format.raw/*671.9*/("""}"""),format.raw/*671.10*/("""
    """),format.raw/*672.5*/("""}"""),format.raw/*672.6*/("""

    """),format.raw/*674.5*/("""//Do a HTTP request on the specified path, parse and insert into the provided element
    function loadCandidateDetails(path, elementToAppendTo)"""),format.raw/*675.59*/("""{"""),format.raw/*675.60*/("""
        """),format.raw/*676.9*/("""$.get(path, function (data) """),format.raw/*676.37*/("""{"""),format.raw/*676.38*/("""
"""),format.raw/*677.1*/("""//            var jsonObj = JSON.parse(JSON.stringify(data));
//            var components = jsonObj['renderableComponents'];
//            var len = (!components ? 0 : components.length);
//            for (var i = 0; i < len; i++) """),format.raw/*680.45*/("""{"""),format.raw/*680.46*/("""
"""),format.raw/*681.1*/("""//                var c = components[i];
//                var temp = createAndAddComponent(c,elementToAppendTo);
//            """),format.raw/*683.15*/("""}"""),format.raw/*683.16*/("""

            """),format.raw/*685.13*/("""var str = JSON.stringify(data);
            var component = Component.getComponent(str);
            component.render(elementToAppendTo);
        """),format.raw/*688.9*/("""}"""),format.raw/*688.10*/(""");
    """),format.raw/*689.5*/("""}"""),format.raw/*689.6*/("""



    """),format.raw/*693.5*/("""//Sorting by column: Intercept click events on table header
    $(function()"""),format.raw/*694.17*/("""{"""),format.raw/*694.18*/("""
        """),format.raw/*695.9*/("""$("#resultsTableHeader").delegate("th", "click", function(e) """),format.raw/*695.70*/("""{"""),format.raw/*695.71*/("""
            """),format.raw/*696.13*/("""//console.log("Header clicked on at: " + $(e.currentTarget).index() + " - " + $(e.currentTarget).html());
            //Update the sort order for the table:
            var clickIndex = $(e.currentTarget).index();
            if(clickIndex == resultTableSortIndex)"""),format.raw/*699.51*/("""{"""),format.raw/*699.52*/("""
                """),format.raw/*700.17*/("""//Switch sort order: ascending -> descending or descending -> ascending
                if(resultTableSortOrder == "ascending")"""),format.raw/*701.56*/("""{"""),format.raw/*701.57*/("""
                    """),format.raw/*702.21*/("""resultTableSortOrder = "descending";
                """),format.raw/*703.17*/("""}"""),format.raw/*703.18*/(""" """),format.raw/*703.19*/("""else """),format.raw/*703.24*/("""{"""),format.raw/*703.25*/("""
                    """),format.raw/*704.21*/("""resultTableSortOrder = "ascending";
                """),format.raw/*705.17*/("""}"""),format.raw/*705.18*/("""
            """),format.raw/*706.13*/("""}"""),format.raw/*706.14*/(""" """),format.raw/*706.15*/("""else """),format.raw/*706.20*/("""{"""),format.raw/*706.21*/("""
                """),format.raw/*707.17*/("""//Sort on column, ascending:
                resultTableSortIndex = clickIndex;
                resultTableSortOrder = "ascending";
            """),format.raw/*710.13*/("""}"""),format.raw/*710.14*/("""

            """),format.raw/*712.13*/("""//Clear record of expanded rows
            expandedRowsCandidateIDs = [];

            //Redraw table
            drawResultTable();
        """),format.raw/*717.9*/("""}"""),format.raw/*717.10*/(""");
    """),format.raw/*718.5*/("""}"""),format.raw/*718.6*/(""");

    //Displaying model/candidate details: Intercept click events on table rows -> toggle visibility on content rows
    $(function()"""),format.raw/*721.17*/("""{"""),format.raw/*721.18*/("""
        """),format.raw/*722.9*/("""$("#resultsTableBody").delegate("tr", "click", function(e)"""),format.raw/*722.67*/("""{"""),format.raw/*722.68*/("""
"""),format.raw/*723.1*/("""//            console.log("Clicked row: " + this.id + " with class: " + this.className);
            var id = this.id;   //Expect: resultTableRow-X  where X is some index
            var dashIdx = id.indexOf("-");
            var candidateID = Number(id.substring(dashIdx+1));
            if(this.className == "resultTableRow")"""),format.raw/*727.51*/("""{"""),format.raw/*727.52*/("""
                """),format.raw/*728.17*/("""var contentRow = $('#' + this.id + '-content');
                var expRowsArrayIdx = expandedRowsCandidateIDs.indexOf(candidateID);
                if(expRowsArrayIdx == -1 )"""),format.raw/*730.43*/("""{"""),format.raw/*730.44*/("""
                    """),format.raw/*731.21*/("""//Currently hidden
                    expandedRowsCandidateIDs.push(candidateID); //Mark as expanded
                    var innerTD = $('#' + this.id + '-content-td');
                    innerTD.empty();
                    var path = "/modelResults/" + candidateID;
                    loadCandidateDetails(path,innerTD);
                """),format.raw/*737.17*/("""}"""),format.raw/*737.18*/(""" """),format.raw/*737.19*/("""else """),format.raw/*737.24*/("""{"""),format.raw/*737.25*/("""
                    """),format.raw/*738.21*/("""//Currently expanded
                    expandedRowsCandidateIDs.splice(expRowsArrayIdx,1);
                """),format.raw/*740.17*/("""}"""),format.raw/*740.18*/("""
                """),format.raw/*741.17*/("""contentRow.toggle();
            """),format.raw/*742.13*/("""}"""),format.raw/*742.14*/("""
        """),format.raw/*743.9*/("""}"""),format.raw/*743.10*/(""");
    """),format.raw/*744.5*/("""}"""),format.raw/*744.6*/(""");

</script>
        <script>
    $(function() """),format.raw/*748.18*/("""{"""),format.raw/*748.19*/("""
        """),format.raw/*749.9*/("""$( "#accordion" ).accordion("""),format.raw/*749.37*/("""{"""),format.raw/*749.38*/("""
            """),format.raw/*750.13*/("""collapsible: true,
            heightStyle: "content"
        """),format.raw/*752.9*/("""}"""),format.raw/*752.10*/(""");
    """),format.raw/*753.5*/("""}"""),format.raw/*753.6*/(""");
    $(function() """),format.raw/*754.18*/("""{"""),format.raw/*754.19*/("""
        """),format.raw/*755.9*/("""$( "#accordion2" ).accordion("""),format.raw/*755.38*/("""{"""),format.raw/*755.39*/("""
            """),format.raw/*756.13*/("""collapsible: true,
            heightStyle: "content"
        """),format.raw/*758.9*/("""}"""),format.raw/*758.10*/(""");
    """),format.raw/*759.5*/("""}"""),format.raw/*759.6*/(""");

</script>

        <table style="width: 100%; padding: 5px;" class="hd">
            <tbody>
                <tr style="height:40px">
                    <td> <div style="width:40px; height:40px; float:left"></div><div style="height:40px; float:left; margin-top: 12px">Arbiter UI</div></td>
                </tr>
            </tbody>
        </table>

        <div style="width:1400px; margin-left:auto; margin-right:auto;">
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
                    <div class="settingsdiv" id="settingsdiv">
            </div>
                </div>
            </div>


            <div class="outerelements" id="results">
                <div class="resultsHeadingDiv">Results</div>
                <div class="resultsdiv" id="resultsdiv">
                    <table style="width:100%" id="resultsTable" class="resultsTable">
                        <col width="33%">
                        <col width="33%">
                        <col width="34%">
                        <thead id="resultsTableHeader"></thead>
                        <tbody id="resultsTableBody"></tbody>
                    </table>
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
                  DATE: Wed Jul 19 14:12:50 AEST 2017
                  SOURCE: C:/DL4J/Git/Arbiter/arbiter-ui/src/main/views/org/deeplearning4j/arbiter/ui/views/ArbiterUI.scala.html
                  HASH: 43f7a6844d52a97b790637846fc92a2e08058418
                  MATRIX: 647->0|1031->356|1060->357|1102->371|1224->466|1253->467|1292->479|1331->490|1360->491|1402->505|1503->579|1532->580|1571->592|1608->601|1637->602|1679->616|1742->652|1771->653|1810->665|1841->668|1870->669|1912->683|2164->908|2193->909|2232->921|2263->924|2292->925|2334->939|2588->1166|2617->1167|2656->1179|2703->1198|2732->1199|2774->1213|2946->1358|2975->1359|3014->1371|3110->1439|3139->1440|3181->1454|3323->1569|3352->1570|3391->1582|3441->1604|3470->1605|3512->1619|3683->1763|3712->1764|3751->1776|3801->1798|3830->1799|3872->1813|4002->1916|4031->1917|4070->1929|4230->2061|4259->2062|4301->2076|4454->2202|4483->2203|4520->2213|4577->2242|4606->2243|4648->2257|4904->2486|4933->2487|4972->2499|5058->2557|5087->2558|5129->2572|5222->2638|5251->2639|5290->2651|5426->2759|5455->2760|5497->2774|5558->2808|5587->2809|5626->2821|5673->2840|5702->2841|5744->2855|6344->3427|6374->3428|6414->3440|6461->3458|6491->3459|6534->3473|6593->3504|6623->3505|6663->3517|6716->3541|6746->3542|6789->3556|6848->3587|6878->3588|6918->3600|7013->3666|7043->3667|7086->3681|7541->4108|7571->4109|7611->4121|7700->4181|7730->4182|7773->4196|7871->4266|7901->4267|7977->4314|8007->4315|8050->4329|8381->4632|8411->4633|8451->4645|8513->4678|8543->4679|8586->4693|8697->4776|8727->4777|8765->4787|8817->4810|8847->4811|8890->4825|9008->4915|9038->4916|9076->4926|9116->4937|9146->4938|9189->4952|9282->5017|9312->5018|9352->5030|10625->6274|10655->6275|10693->6285|10816->6379|10846->6380|10889->6394|11505->6981|11535->6982|11582->7000|11708->7097|11738->7098|11789->7120|12101->7403|12131->7404|12230->7474|12260->7475|12305->7491|12429->7586|12459->7587|12506->7605|12624->7694|12654->7695|12705->7717|13000->7983|13030->7984|13133->8058|13163->8059|13208->8075|13363->8201|13393->8202|13442->8222|13569->8320|13599->8321|13650->8343|13840->8504|13870->8505|13971->8577|14001->8578|14039->8588|14069->8589|14104->8596|14133->8597|14264->8699|14294->8700|14332->8710|14494->8843|14524->8844|15308->9600|15338->9601|15372->9607|15401->9608|15437->9616|15513->9663|15543->9664|15581->9674|15838->9902|15868->9903|15911->9917|15987->9965|16017->9966|16057->9978|16288->10180|16318->10181|16361->10195|16494->10299|16524->10300|16571->10318|16664->10382|16694->10383|16737->10397|16842->10474|16872->10475|16912->10487|16951->10497|16981->10498|17024->10512|17081->10540|17111->10541|17158->10559|17340->10712|17370->10713|17421->10735|17511->10796|17541->10797|17588->10815|17690->10888|17720->10889|17758->10899|17788->10900|17828->10912|17923->10979|17952->10980|17988->10988|18240->11211|18270->11212|18308->11222|18709->11594|18739->11595|18808->11635|18838->11636|19585->12354|19615->12355|19645->12356|19697->12379|19727->12380|19790->12414|19820->12415|19850->12416|19902->12439|19932->12440|20608->13087|20638->13088|20681->13102|21036->13429|21066->13430|21104->13440|21349->13656|21379->13657|21422->13671|21558->13778|21588->13779|21635->13797|21671->13804|21701->13805|21731->13806|21797->13843|21827->13844|21871->13859|21901->13860|22115->14046|22145->14047|22185->14059|22586->14431|22616->14432|22659->14446|22764->14522|22794->14523|22841->14541|23647->15318|23677->15319|23715->15329|23745->15330|23785->15342|23857->15385|23887->15386|23930->15400|24286->15728|24316->15729|24350->15735|24379->15736|24415->15744|24697->15997|24727->15998|24765->16008|25229->16443|25259->16444|25328->16484|25358->16485|26105->17203|26135->17204|26165->17205|26217->17228|26247->17229|26310->17263|26340->17264|26370->17265|26422->17288|26452->17289|27128->17936|27158->17937|27201->17951|27556->18278|27586->18279|27624->18289|27869->18505|27899->18506|27942->18520|28078->18627|28108->18628|28155->18646|28191->18653|28221->18654|28251->18655|28317->18692|28347->18693|28391->18708|28421->18709|28641->18900|28671->18901|28701->18902|28745->18917|28775->18918|28886->19000|28916->19001|28946->19002|29001->19028|29031->19029|29106->19075|29136->19076|29166->19077|29221->19103|29251->19104|29291->19116|29321->19117|29361->19129|29762->19501|29792->19502|29835->19516|29940->19592|29970->19593|30017->19611|30823->20388|30853->20389|30891->20399|30921->20400|30961->20412|31033->20455|31063->20456|31106->20470|31462->20798|31492->20799|31526->20805|31555->20806|31591->20814|31669->20863|31699->20864|31737->20874|32290->21398|32320->21399|32363->21413|32428->21449|32458->21450|32544->21506|32575->21507|32615->21519|32645->21520|32675->21521|32709->21526|32739->21527|32782->21541|32847->21577|32877->21578|32947->21619|32977->21620|33017->21632|33047->21633|33087->21645|33311->21840|33341->21841|33384->21855|33528->21971|33558->21972|33592->21978|33621->21979|33657->21987|33712->22013|33742->22014|33782->22026|35139->23354|35169->23355|35212->23369|36123->24251|36153->24252|36200->24270|36263->24304|36293->24305|36323->24306|36357->24311|36387->24312|36434->24330|36742->24609|36772->24610|36810->24620|36840->24621|36874->24627|36903->24628|36939->24636|37059->24727|37089->24728|37127->24738|37246->24829|37275->24830|37309->24836|37365->24863|37395->24864|37433->24874|37539->24951|37569->24952|37612->24966|37675->25001|37705->25002|37735->25003|37769->25008|37799->25009|37842->25023|37905->25058|37935->25059|37969->25065|37998->25066|38032->25072|38088->25099|38118->25100|38156->25110|38276->25201|38306->25202|38349->25216|38453->25292|38483->25293|38513->25294|38547->25299|38577->25300|38620->25314|38724->25390|38754->25391|38788->25397|38817->25398|38853->25406|39027->25551|39057->25552|39095->25562|39152->25590|39182->25591|39212->25593|39477->25829|39507->25830|39537->25832|39696->25962|39726->25963|39771->25979|39948->26128|39978->26129|40014->26137|40043->26138|40083->26150|40189->26227|40219->26228|40257->26238|40347->26299|40377->26300|40420->26314|40716->26581|40746->26582|40793->26600|40950->26728|40980->26729|41031->26751|41114->26805|41144->26806|41174->26807|41208->26812|41238->26813|41289->26835|41371->26888|41401->26889|41444->26903|41474->26904|41504->26905|41538->26910|41568->26911|41615->26929|41791->27076|41821->27077|41866->27093|42041->27240|42071->27241|42107->27249|42136->27250|42304->27389|42334->27390|42372->27400|42459->27458|42489->27459|42519->27461|42879->27792|42909->27793|42956->27811|43162->27988|43192->27989|43243->28011|43620->28359|43650->28360|43680->28361|43714->28366|43744->28367|43795->28389|43935->28500|43965->28501|44012->28519|44075->28553|44105->28554|44143->28564|44173->28565|44209->28573|44238->28574|44319->28626|44349->28627|44387->28637|44444->28665|44474->28666|44517->28680|44609->28744|44639->28745|44675->28753|44704->28754|44754->28775|44784->28776|44822->28786|44880->28815|44910->28816|44953->28830|45045->28894|45075->28895|45111->28903|45140->28904
                  LINES: 25->1|36->12|36->12|37->13|40->16|40->16|42->18|42->18|42->18|43->19|46->22|46->22|48->24|48->24|48->24|49->25|50->26|50->26|52->28|52->28|52->28|53->29|59->35|59->35|61->37|61->37|61->37|62->38|68->44|68->44|70->46|70->46|70->46|71->47|75->51|75->51|77->53|77->53|77->53|78->54|80->56|80->56|82->58|82->58|82->58|83->59|87->63|87->63|89->65|89->65|89->65|90->66|93->69|93->69|95->71|96->72|96->72|97->73|99->75|99->75|100->76|100->76|100->76|101->77|106->82|106->82|108->84|109->85|109->85|110->86|112->88|112->88|114->90|115->91|115->91|116->92|117->93|117->93|119->95|119->95|119->95|120->96|136->112|136->112|138->114|138->114|138->114|139->115|140->116|140->116|142->118|142->118|142->118|143->119|144->120|144->120|146->122|146->122|146->122|147->123|155->131|155->131|157->133|158->134|158->134|159->135|161->137|161->137|163->139|163->139|164->140|170->146|170->146|172->148|173->149|173->149|174->150|177->153|177->153|178->154|178->154|178->154|179->155|182->158|182->158|183->159|183->159|183->159|184->160|186->162|186->162|188->164|217->193|217->193|218->194|219->195|219->195|220->196|229->205|229->205|230->206|231->207|231->207|232->208|238->214|238->214|241->217|241->217|243->219|244->220|244->220|245->221|246->222|246->222|247->223|254->230|254->230|257->233|257->233|259->235|260->236|260->236|262->238|263->239|263->239|264->240|267->243|267->243|270->246|270->246|271->247|271->247|272->248|272->248|274->250|274->250|275->251|278->254|278->254|297->273|297->273|298->274|298->274|300->276|300->276|300->276|301->277|307->283|307->283|308->284|309->285|309->285|311->287|314->290|314->290|315->291|317->293|317->293|318->294|319->295|319->295|320->296|322->298|322->298|324->300|324->300|324->300|325->301|325->301|325->301|326->302|329->305|329->305|330->306|331->307|331->307|332->308|334->310|334->310|335->311|335->311|337->313|339->315|339->315|341->317|344->320|344->320|345->321|354->330|354->330|354->330|354->330|373->349|373->349|373->349|373->349|373->349|374->350|374->350|374->350|374->350|374->350|390->366|390->366|391->367|399->375|399->375|400->376|406->382|406->382|407->383|410->386|410->386|411->387|411->387|411->387|411->387|411->387|411->387|412->388|412->388|417->393|417->393|419->395|431->407|431->407|432->408|433->409|433->409|434->410|448->424|448->424|449->425|449->425|451->427|452->428|452->428|453->429|460->436|460->436|461->437|461->437|463->439|466->442|466->442|467->443|477->453|477->453|477->453|477->453|496->472|496->472|496->472|496->472|496->472|497->473|497->473|497->473|497->473|497->473|513->489|513->489|514->490|522->498|522->498|523->499|529->505|529->505|530->506|533->509|533->509|534->510|534->510|534->510|534->510|534->510|534->510|535->511|535->511|541->517|541->517|541->517|541->517|541->517|543->519|543->519|543->519|543->519|543->519|544->520|544->520|544->520|544->520|544->520|545->521|545->521|547->523|559->535|559->535|560->536|561->537|561->537|562->538|576->552|576->552|577->553|577->553|579->555|580->556|580->556|581->557|588->564|588->564|589->565|589->565|591->567|591->567|591->567|592->568|604->580|604->580|605->581|605->581|605->581|605->581|605->581|606->582|606->582|606->582|606->582|606->582|607->583|607->583|607->583|607->583|607->583|608->584|608->584|610->586|613->589|613->589|614->590|616->592|616->592|617->593|617->593|619->595|619->595|619->595|621->597|646->622|646->622|647->623|662->638|662->638|663->639|665->641|665->641|665->641|665->641|665->641|666->642|673->649|673->649|674->650|674->650|675->651|675->651|677->653|678->654|678->654|679->655|680->656|680->656|681->657|681->657|681->657|682->658|683->659|683->659|684->660|685->661|685->661|685->661|685->661|685->661|686->662|687->663|687->663|688->664|688->664|689->665|689->665|689->665|690->666|691->667|691->667|692->668|693->669|693->669|693->669|693->669|693->669|694->670|695->671|695->671|696->672|696->672|698->674|699->675|699->675|700->676|700->676|700->676|701->677|704->680|704->680|705->681|707->683|707->683|709->685|712->688|712->688|713->689|713->689|717->693|718->694|718->694|719->695|719->695|719->695|720->696|723->699|723->699|724->700|725->701|725->701|726->702|727->703|727->703|727->703|727->703|727->703|728->704|729->705|729->705|730->706|730->706|730->706|730->706|730->706|731->707|734->710|734->710|736->712|741->717|741->717|742->718|742->718|745->721|745->721|746->722|746->722|746->722|747->723|751->727|751->727|752->728|754->730|754->730|755->731|761->737|761->737|761->737|761->737|761->737|762->738|764->740|764->740|765->741|766->742|766->742|767->743|767->743|768->744|768->744|772->748|772->748|773->749|773->749|773->749|774->750|776->752|776->752|777->753|777->753|778->754|778->754|779->755|779->755|779->755|780->756|782->758|782->758|783->759|783->759
                  -- GENERATED --
              */
          