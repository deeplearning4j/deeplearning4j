<!DOCTYPE html>
<html>
<head>
    <style type="text/css">
        /* Color and style reference.
            To change: do find + replace on comment + color

            heading background:         headingbgcol        #063E53            //Old candidates: #3B5998
            heading text color:         headingtextcolor    white

        */

        html, body {
            width: 100%;
            height: 100%;
            padding-top: 20px;
            padding-left: 20px;
            padding-right: 20px;
            padding-bottom: 20px;
        }

        .bgcolor {
            background-color: #D1C5B7; /* OLD: #D2E4EF;*/
        }

        h1 {
            font-family: Georgia, Times, 'Times New Roman', serif;
            font-size: 28px;
            font-style: bold;
            font-variant: normal;
            font-weight: 500;
            line-height: 26.4px;
        }

        h3 {
            font-family: Georgia, Times, 'Times New Roman', serif;
            font-size: 16px;
            font-style: normal;
            font-variant: normal;
            font-weight: 500;
            line-height: 26.4px;
        }

        table.resultsTable {
            border-collapse:collapse;
            background-color: white;
            /*border-collapse: collapse;*/
            padding: 15px;
        }

        table.resultsTable td, table.resultsTable tr, table.resultsTable th {
            border:solid black 1px;
            white-space: pre;   /* assume text is preprocessed for formatting */
        }

        table.resultsTable th {
            background-color: /*headingbgcol*/#063E53;
            color: white;
            padding-left: 4px;
            padding-right: 4px;
        }

        table.resultsTable td {
            /*background-color: white;*/
            padding-left: 4px;
            padding-right: 4px;
        }

        /* Properties for table cells in the tables generated using the RenderableComponent mechanism */
        .renderableComponentTable {
            /*table-layout:fixed; */    /*Avoids scrollbar, but makes fixed width for all columns :( */
            width: 100%
        }
        .renderableComponentTable td {
            padding-left: 4px;
            padding-right: 4px;
            white-space: pre;   /* assume text is pre-processed (important for line breaks etc)*/
            word-wrap:break-word;
            vertical-align: top;
        }

        /** CSS for result table rows */
        .resultTableRow {
            background-color: #E1E8EA; /*#D7E9EF;*/
            cursor: pointer;
        }

        /** CSS for result table CONTENT rows (i.e., only visible when expanded) */
        .resultTableRowContent {
            background-color: white;
        }

        .resultsHeadingDiv {
            background-color: /*headingbgcol*/#063E53;
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
        }

        div.outerelements {
            padding-bottom: 30px;
        }

        #accordion, #accordion2 {
            padding-bottom: 20px;
        }

        #accordion .ui-accordion-header, #accordion2 .ui-accordion-header {
            background-color: /*headingbgcolor*/#063E53;      /*Color when collapsed*/
            color: /*headingtextcolor*/white;
            font-family: Georgia, Times, 'Times New Roman', serif;
            font-size: 20px;
            font-style: bold;
            font-variant: normal;
            margin: 0px;
            background-image: none;     /* Necessary, otherwise color changes don't make a difference */
        }

        /*
        #accordion .ui-accordion-header.ui-state-active {
            background-color: pink;
            background-image: none;
        }*/

        #accordion .ui-accordion-content {
            width: 100%;
            background-color: white;    /*background color of accordian content (elements in front may have different color */
            color: black;  /* text etc color */
            font-size: 10pt;
            line-height: 16pt;
            overflow:visible !important;
        }

        /** Line charts */
        path {
            stroke: steelblue;
            stroke-width: 2;
            fill: none;
        }
        .axis path, .axis line {
            fill: none;
            stroke: #000;
            shape-rendering: crispEdges;
        }
        .tick line {
            opacity: 0.2;
            shape-rendering: crispEdges;
        }

    </style>
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
    setInterval(function(){
        //Get the update status, and do something with it:
        $.get("/lastUpdate",function(data){
            //Encoding: matches names in UpdateStatus class
            var jsonObj = JSON.parse(JSON.stringify(data));
            var statusTime = jsonObj['statusUpdateTime'];
            var settingsTime = jsonObj['settingsUpdateTime'];
            var resultsTime = jsonObj['resultsUpdateTime'];
            //console.log("Last update times: " + statusTime + ", " + settingsTime + ", " + resultsTime);

            //Check last update times for each part of document, and update as necessary
            //First section: summary status
            if(lastStatusUpdateTime != statusTime){
                //Get JSON: address set by SummaryStatusResource
                $.get("/summary",function(data){
                    var jsonObj = JSON.parse(JSON.stringify(data));

                    var summaryStatusDiv = $('#statusdiv');
                    var components = jsonObj['renderableComponents'];
                    if(!components) summaryStatusDiv.html('');
                    summaryStatusDiv.html('');

                    var len = (!components ? 0 : components.length);
                    for(var i=0; i<len; i++){
                        var c = components[i];
                        createAndAddComponent(c,summaryStatusDiv);
                    }
                });

                lastStatusUpdateTime = statusTime;
            }

            //Second section: Optimization settings
            if(lastSettingsUpdateTime != settingsTime){
                //Get JSON: address set by ConfigResource
                $.get("/config",function(data){
                    var jsonObj = JSON.parse(JSON.stringify(data));

                    var components = jsonObj['renderableComponents'];

                    var configDiv = $('#settingsdiv');
                    configDiv.html('');

                    var len = (!components ? 0 : components.length);
                    for(var i=0; i<len; i++){
                        var c = components[i];
                        createAndAddComponent(c,configDiv);
                    }
                });

                lastSettingsUpdateTime = settingsTime;
            }

            //Third section: Summary results table (summary info for each candidate)
            if(lastResultsUpdateTime != resultsTime){

                //Get JSON; address set by SummaryResultsResource
                $.get("/results",function(data){
                    //Expect an array of CandidateStatus type objects here
                    resultsTableContent = data;
                    drawResultTable();
                });

                lastResultsUpdateTime = resultsTime;
            }
        })
    },2000);    //Loop every 2 seconds

    function createAndAddComponent(renderableComponent, appendTo){
        var key = Object.keys(renderableComponent)[0];
        var type = renderableComponent[key]['componentType'];

        switch(type){
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
        }
    }

    function createTable(tableObj,tableId,appendTo){
        //Expect RenderableComponentTable
        var header = tableObj['header'];
        var values = tableObj['table'];
        var title = tableObj['title'];
        var nRows = (values ? values.length : 0);

        if(title){
            appendTo.append("<h5>"+title+"</h5>");
        }

        var table;
        if(tableId) table = $("<table id=\"" + tableId + "\" class=\"renderableComponentTable\">");
        else table = $("<table class=\"renderableComponentTable\">");
        if(header){
            var headerRow = $("<tr>");
            var len = header.length;
            for( var i=0; i<len; i++ ){
                headerRow.append($("<th>" + header[i] + "</th>"));
            }
            headerRow.append($("</tr>"));
            table.append(headerRow);
        }

        if(values){
            for( var i=0; i<nRows; i++ ){
                var row = $("<tr>");
                var rowValues = values[i];
                var len = rowValues.length;
                for( var j=0; j<len; j++ ){
                    row.append($('<td>'+rowValues[j]+'</td>'));
                }
                row.append($("</tr>"));
                table.append(row);
            }
        }

        table.append($("</table>"));
        appendTo.append(table);
    }

    /** Create + add line chart with multiple lines, (optional) title, (optional) series names.
     * appendTo: jquery selector of object to append to. MUST HAVE ID
     * */
    function createLineChart(chartObj, appendTo){
        //Expect: RenderableComponentLineChart
        var title = chartObj['title'];
        var xData = chartObj['x'];
        var yData = chartObj['y'];
        var seriesNames = chartObj['seriesNames'];
        var nSeries = (!xData ? 0 : xData.length);
        var title = chartObj['title'];

        // Set the dimensions of the canvas / graph
        var margin = {top: 60, right: 20, bottom: 60, left: 50},
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
                .x(function(d) { return xScale(d.xPos); })
                .y(function(d) { return yScale(d.yPos); });

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
        for( var i=0; i<nSeries; i++){
            var xV = xData[i];
            var yV = yData[i];
            var thisMax = d3.max(xV);
            var thisMaxY = d3.max(yV);
            var thisMinY = d3.min(yV);
            if(thisMax > xMax) xMax = thisMax;
            if(thisMaxY > yMax) yMax = thisMaxY;
            if(thisMinY < yMin) yMin = thisMinY;
        }
        if(yMin > 0) yMin = 0;
        xScale.domain([0, xMax]);
        yScale.domain([yMin, yMax]);

        // Add the valueline path.
        var color = d3.scale.category10();
        for( var i=0; i<nSeries; i++){
            var xVals = xData[i];
            var yVals = yData[i];

            var data = xVals.map(function(d, i){
                return { 'xPos' : xVals[i], 'yPos' : yVals[i] };
            });
            svg.append("path")
                    .attr("class", "line")
                    .style("stroke", color(i))
                    .attr("d", valueline(data));
        }

        // Add the X Axis
        svg.append("g")
                .attr("class", "x axis")
                .attr("transform", "translate(0," + height + ")")
                .call(xAxis);

        // Add the Y Axis
        svg.append("g")
                .attr("class", "y axis")
                .call(yAxis);

        //Add legend (if present)
        if(seriesNames) {
            var legendSpace = width / i;
            for (var i = 0; i < nSeries; i++) {
                var values = xData[i];
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

            }
        }

        //Add title (if present)
        if(title){
            svg.append("text")
                    .attr("x", (width / 2))
                    .attr("y", 0 - ((margin.top-30) / 2))
                    .attr("text-anchor", "middle")
                    .style("font-size", "13px")
                    .style("text-decoration", "underline")
                    .text(title);
        }
    }

    /** Create + add scatter plot chart with multiple different types of points, (optional) title, (optional) series names.
     * appendTo: jquery selector of object to append to. MUST HAVE ID
     * */
    function createScatterPlot(chartObj, appendTo){
        //TODO modify this to do scatter plot, not line chart
        //Expect: RenderableComponentLineChart
        var title = chartObj['title'];
        var xData = chartObj['x'];
        var yData = chartObj['y'];
        var seriesNames = chartObj['seriesNames'];
        var nSeries = (!xData ? 0 : xData.length);
        var title = chartObj['title'];

        // Set the dimensions of the canvas / graph
        var margin = {top: 60, right: 20, bottom: 60, left: 50},
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
                .x(function(d) { return xScale(d.xPos); })
                .y(function(d) { return yScale(d.yPos); });

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
        for( var i=0; i<nSeries; i++){
            var xV = xData[i];
            var yV = yData[i];
            var thisMax = d3.max(xV);
            var thisMaxY = d3.max(yV);
            var thisMinY = d3.min(yV);
            if(thisMax > xMax) xMax = thisMax;
            if(thisMaxY > yMax) yMax = thisMaxY;
            if(thisMinY < yMin) yMin = thisMinY;
        }
        if(yMin > 0) yMin = 0;
        xScale.domain([0, xMax]);
        yScale.domain([yMin, yMax]);

        // Add the valueline path.
        var color = d3.scale.category10();
        for( var i=0; i<nSeries; i++){
            var xVals = xData[i];
            var yVals = yData[i];

            var data = xVals.map(function(d, i){
                return { 'xPos' : xVals[i], 'yPos' : yVals[i] };
            });

            svg.selectAll("circle")
                    .data(data)
                    .enter()
                    .append("circle")
                    .style("fill", function(d){ return color(i)})
                    .attr("r",3.0)
                    .attr("cx", function(d){ return xScale(d['xPos']); })
                    .attr("cy", function(d){ return yScale(d['yPos']); });
        }

        // Add the X Axis
        svg.append("g")
                .attr("class", "x axis")
                .attr("transform", "translate(0," + height + ")")
                .call(xAxis);

        // Add the Y Axis
        svg.append("g")
                .attr("class", "y axis")
                .call(yAxis);

        //Add legend (if present)
        if(seriesNames) {
            var legendSpace = width / i;
            for (var i = 0; i < nSeries; i++) {
                var values = xData[i];
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

            }
        }

        //Add title (if present)
        if(title){
            svg.append("text")
                    .attr("x", (width / 2))
                    .attr("y", 0 - ((margin.top-30) / 2))
                    .attr("text-anchor", "middle")
                    .style("font-size", "13px")
                    .style("text-decoration", "underline")
                    .text(title);
        }
    }

    function createAccordion(accordionObj, appendTo) {
        var title = accordionObj['title'];
        var defaultCollapsed = accordionObj['defaultCollapsed'];

        var tempDivOuter = $('<div><h3>' + title + '</h3></div>');
        tempDivOuter.uniqueId();
        var generatedIDOuter = tempDivOuter.attr('id');
        var tempDivInner = $('<div></div>');
        tempDivInner.uniqueId();
        var generatedIDInner = tempDivInner.attr('id');
        tempDivOuter.append(tempDivInner);
        appendTo.append(tempDivOuter);

        if (defaultCollapsed == true) {
            $("#" + generatedIDOuter).accordion({collapsible: true, heightStyle: "content", active: false});
        } else {
            $("#" + generatedIDOuter).accordion({collapsible: true, heightStyle: "content"});
        }

        //Add the inner components:
        var innerComponents = accordionObj['innerComponents'];
        var len = (!innerComponents ? 0 : innerComponents.length);
        for( var i=0; i<len; i++ ){
            var component = innerComponents[i];
            createAndAddComponent(component,$("#"+generatedIDInner));
        }
    }

    function drawResultTable(){

        //Remove all elements from the table body
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
        for(var i=0; i<len; i++){
            var row = $('<tr class="resultTableRow" id="resultTableRow-' + sorted[i].index + '"/>');
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
            if(expandedRowsCandidateIDs.indexOf(sorted[i].index) == -1 ){
                contentRow.hide();

            } else {
                //Load info. TODO: make this more efficient (stored info, check for updates, etc)
                td3.empty();

                var path = "/modelResults/" + sorted[i].index;
                loadCandidateDetails(path, td3);

                contentRow.show();
            }
        }
    }

    //Compare function for results, based on sort order
    function compareResultsIndex(a, b){
        return (resultTableSortOrder == "ascending" ? a.index - b.index : b.index - a.index);
    }
    function compareScores(a,b){
        //TODO Not always numbers...
        if(resultTableSortOrder == "ascending"){
            return a.score - b.score;
        } else {
            return b.score - a.score;
        }
    }
    function compareStatus(a,b){
        //TODO: secondary sort on... score? index?
        if(resultTableSortOrder == "ascending"){
            return (a.status < b.status ? -1 : (a.status > b.status ? 1 : 0));
        } else {
            return (a.status < b.status ? 1 : (a.status > b.status ? -1 : 0));
        }
    }

    //Do a HTTP request on the specified path, parse and insert into the provided element
    function loadCandidateDetails(path, elementToAppendTo){
        $.get(path, function (data) {
            var jsonObj = JSON.parse(JSON.stringify(data));
            var components = jsonObj['renderableComponents'];
            var len = (!components ? 0 : components.length);
            for (var i = 0; i < len; i++) {
                var c = components[i];
                var temp = createAndAddComponent(c,elementToAppendTo);
            }
        });
    }



    //Sorting by column: Intercept click events on table header
    $(function(){
        $("#resultsTableHeader").delegate("th", "click", function(e) {
            //console.log("Header clicked on at: " + $(e.currentTarget).index() + " - " + $(e.currentTarget).html());
            //Update the sort order for the table:
            var clickIndex = $(e.currentTarget).index();
            if(clickIndex == resultTableSortIndex){
                //Switch sort order: ascending -> descending or descending -> ascending
                if(resultTableSortOrder == "ascending"){
                    resultTableSortOrder = "descending";
                } else {
                    resultTableSortOrder = "ascending";
                }
            } else {
                //Sort on column, ascending:
                resultTableSortIndex = clickIndex;
                resultTableSortOrder = "ascending";
            }

            //Clear record of expanded rows
            expandedRowsCandidateIDs = [];

            //Redraw table
            drawResultTable();
        });
    });

    //Displaying model/candidate details: Intercept click events on table rows -> toggle visibility on content rows
    $(function(){
        $("#resultsTableBody").delegate("tr", "click", function(e){
//            console.log("Clicked row: " + this.id + " with class: " + this.className);
            var id = this.id;   //Expect: resultTableRow-X  where X is some index
            var dashIdx = id.indexOf("-");
            var candidateID = Number(id.substring(dashIdx+1));
            if(this.className == "resultTableRow"){
                var contentRow = $('#' + this.id + '-content');
                var expRowsArrayIdx = expandedRowsCandidateIDs.indexOf(candidateID);
                if(expRowsArrayIdx == -1 ){
                    //Currently hidden
                    expandedRowsCandidateIDs.push(candidateID); //Mark as expanded
                    var innerTD = $('#' + this.id + '-content-td');
                    innerTD.empty();
                    var path = "/modelResults/" + candidateID;
                    loadCandidateDetails(path,innerTD);
                } else {
                    //Currently expanded
                    expandedRowsCandidateIDs.splice(expRowsArrayIdx,1);
                }
                contentRow.toggle();
            }
        });
    });

</script>
<script>
    $(function() {
        $( "#accordion" ).accordion({
            collapsible: true,
            heightStyle: "content"
        });
    });
    $(function() {
        $( "#accordion2" ).accordion({
            collapsible: true,
            heightStyle: "content"
        });
    });

</script>




<div class="outerelements" id="heading">
    <h1>Arbiter</h1>
</div>


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
            <thead id="resultsTableHeader"></thead>
            <tbody id="resultsTableBody"></tbody>
        </table>
    </div>
</div>


</body>
</html>