<html>
<head>
    <style type="text/css">
        html, body {
            width: 100%;
            height: 100%;
        }

        .bgcolor {
            background-color: #FFFFFF;
        }

        .hd {
            background-color: #000000;
            font-size: 18px;
            color: #FFFFFF;
        }

        .sectionheader {
            background-color: #888888;
            width:100%;
            font-size: 16px;
            font-style: bold;
            color: #FFFFFF;
            /*padding-left: 40px;*/
            /*padding-right: 8px;*/
            /*padding-top: 2px;*/
            /*padding-bottom: 2px;*/

        }

        .subsectiontop {
            background-color: #F5F5FF;
            height: 300px;
        }

        .subsectionbottom {
            background-color: #F5F5FF;
            height: 540px;
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

        div.outerelements {
            padding-bottom: 30px;
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

        /** Bar charts */
        .bar {
            fill: steelblue;
        }

        rect {
            fill: steelblue;
        }

        .legend rect {
            fill:white;
            stroke:black;
            opacity:0.8;
        }

    </style>
    <title>Data Analysis</title>

</head>
<body style="padding: 0px; margin: 0px" onload="generateContent()">

<#--<meta name="viewport" content="width=device-width, initial-scale=1">-->
<link href="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap.min.css">
<script src="http://ajax.googleapis.com/ajax/libs/jquery/1/jquery.min.js"></script>
<link href="http://code.jquery.com/ui/1.11.4/themes/smoothness/jquery-ui.css">
<script src="http://code.jquery.com/jquery-1.10.2.js"></script>
<script src="http://code.jquery.com/ui/1.11.4/jquery-ui.js"></script>
<script src="http://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js"></script>
<script src="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/js/bootstrap.min.js"></script>
<script>

    function generateContent(){
        var mainDiv = $('#maindiv');

        var div2 = $('#tablesource');
        console.log(div2.html());

        var div2html = div2.html();
        createTable(jQuery.parseJSON(div2html)["RenderableComponentTable"], "Summary Table", $('#tablediv'));

        var histdiv = $("#histogramdiv");

        <#list histogramIDs as histid>
            var div_${histid} = $('#${histid}');
            var html_${histid} = div_${histid}.html();
            createHistogram(jQuery.parseJSON(html_${histid})["RenderableComponentHistogram"], histdiv, 700, 400);

        </#list>

    }

    function createTable(tableObj, tableId, appendTo) {
        //Expect RenderableComponentTable
        var header = tableObj['header'];
        var values = tableObj['table'];
        var title = tableObj['title'];
        var border = tableObj['border'];
        var padLeft = tableObj['padLeftPx'];
        var padRight = tableObj['padRightPx'];
        var padTop = tableObj['padTopPx'];
        var padBottom = tableObj['padBottomPx'];
        var colWidths = tableObj['colWidthsPercent'];
        var nRows = (values ? values.length : 0);
        var backgroundColor = tableObj['backgroundColor'];
        var headerColor = tableObj['headerColor'];


        var tbl = document.createElement('table');
        tbl.style.width = '100%';
//        tbl.style.height = '100%';
        tbl.setAttribute('border', border);
        if(backgroundColor) tbl.style.backgroundColor = backgroundColor;

        if (colWidths) {
            for (var i = 0; i < colWidths.length; i++) {
                var col = document.createElement('col');
                col.setAttribute('width', colWidths[i] + '%');
                tbl.appendChild(col);
            }
        }

        if (header) {
            var theader = document.createElement('thead');
            var headerRow = document.createElement('tr');

            if(headerColor) headerRow.style.backgroundColor = headerColor;

            for (var i = 0; i < header.length; i++) {
                var headerd = document.createElement('th');
                headerd.style.padding = padTop + 'px ' + padRight + 'px ' + padBottom + 'px ' + padLeft + 'px';
                headerd.appendChild(document.createTextNode(header[i]));
                headerRow.appendChild(headerd);
            }
            tbl.appendChild(headerRow);
        }

        //Add content:
        if (values) {

            var tbdy = document.createElement('tbody');
            for (var i = 0; i < values.length; i++) {
                var tr = document.createElement('tr');

                for (var j = 0; j < values[i].length; j++) {
                    var td = document.createElement('td');
                    td.style.padding = padTop + 'px ' + padRight + 'px ' + padBottom + 'px ' + padLeft + 'px';
                    td.appendChild(document.createTextNode(values[i][j]));
                    tr.appendChild(td);
                }

                tbdy.appendChild(tr);
            }
            tbl.appendChild(tbdy);
        }

        appendTo.append(tbl);
    }

    /** Create + add line chart with multiple lines, (optional) title, (optional) series names.
     * appendTo: jquery selector of object to append to. MUST HAVE ID
     * */
    function createLineChart(chartObj, appendTo, chartWidth, chartHeight) {
        //Expect: RenderableComponentLineChart
        var title = chartObj['title'];
        var xData = chartObj['x'];
        var yData = chartObj['y'];
        var mTop = chartObj['marginTop'];
        var mBottom = chartObj['marginBottom'];
        var mLeft = chartObj['marginLeft'];
        var mRight = chartObj['marginRight'];
        var removeAxisHorizontal = chartObj['removeAxisHorizontal'];
        var seriesNames = chartObj['seriesNames'];
        var withLegend = chartObj['legend'];
        var nSeries = (!xData ? 0 : xData.length);

        // Set the dimensions of the canvas / graph
        var margin = {top: mTop, right: mRight, bottom: mBottom, left: mLeft},
                width = chartWidth - margin.left - margin.right,
                height = chartHeight - margin.top - margin.bottom;

        // Set the ranges
        var xScale = d3.scale.linear().range([0, width]);
        var yScale = d3.scale.linear().range([height, 0]);

        // Define the axes
        var xAxis = d3.svg.axis().scale(xScale)
                .innerTickSize(-height)     //used as grid line
                .orient("bottom").ticks(5);

        if(removeAxisHorizontal == true){
            xAxis.tickValues([]);
        }

        var yAxis = d3.svg.axis().scale(yScale)
                .innerTickSize(-width)      //used as grid line
                .orient("left").ticks(5);

        // Define the line
        var valueline = d3.svg.line()
                .x(function (d) {
                    return xScale(d.xPos);
                })
                .y(function (d) {
                    return yScale(d.yPos);
                });

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
        var xMin = Number.MAX_VALUE;
        var xMax = -Number.MAX_VALUE;
        var yMax = -Number.MAX_VALUE;
        var yMin = Number.MAX_VALUE;
        for (var i = 0; i < nSeries; i++) {
            var xV = xData[i];
            var yV = yData[i];
            var thisMin = d3.min(xV);
            var thisMax = d3.max(xV);
            var thisMaxY = d3.max(yV);
            var thisMinY = d3.min(yV);
            if (thisMin < xMin) xMin = thisMin;
            if (thisMax > xMax) xMax = thisMax;
            if (thisMaxY > yMax) yMax = thisMaxY;
            if (thisMinY < yMin) yMin = thisMinY;
        }
        if (yMin > 0) yMin = 0;
        xScale.domain([xMin, xMax]);
        yScale.domain([yMin, yMax]);

        // Add the valueline path.
        var color = d3.scale.category10();
        for (var i = 0; i < nSeries; i++) {
            var xVals = xData[i];
            var yVals = yData[i];

            var data = xVals.map(function (d, i) {
                return {'xPos': xVals[i], 'yPos': yVals[i]};
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
        if (seriesNames && withLegend == true) {
            var legendSpace = width / i;
            for (var i = 0; i < nSeries; i++) {
                var values = xData[i];
                var yValues = yData[i];
                var lastX = values[values.length - 1];
                var lastY = yValues[yValues.length - 1];
                var toDisplay;
                if (!lastX || !lastY) toDisplay = seriesNames[i] + " (no data)";
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
        if (title) {
            svg.append("text")
                    .attr("x", (width / 2))
                    .attr("y", 0 - ((margin.top - 30) / 2))
                    .attr("text-anchor", "middle")
                    .style("font-size", "13px")
                    .style("text-decoration", "underline")
                    .text(title);
        }
    }

    /** Create + add histogram
     * */
    function createHistogram(chartObj, appendTo, chartWidth, chartHeight) {
        //Expect: RenderableComponentHistogram
        var title = chartObj['title'];
        var lowerBounds = chartObj['lowerBounds'];
        var upperBounds = chartObj['upperBounds'];
        var yValues = chartObj['yvalues'];
        var mTop = chartObj['marginTop'];
        var mBottom = chartObj['marginBottom'];
        var mLeft = chartObj['marginLeft'];
        var mRight = chartObj['marginRight'];
//        var removeAxisHorizontal = chartObj['removeAxisHorizontal'];

        // Set the dimensions of the canvas / graph
        var margin = {top: mTop, right: mRight, bottom: mBottom, left: mLeft},
                width = chartWidth - margin.left - margin.right,
                height = chartHeight - margin.top - margin.bottom;

        // Set the ranges
        var xScale = d3.scale.linear().range([0, width]);
        var yScale = d3.scale.linear().range([height, 0]);

        var xMin = Number.MAX_VALUE;
        var xMax = -Number.MAX_VALUE;
        var yMax = -Number.MAX_VALUE;
        for (var i = 0; i < lowerBounds.length; i++) {
            if (lowerBounds[i] < xMin) xMin = lowerBounds[i];
            if (upperBounds[i] > xMax) xMax = upperBounds[i];
            if (yValues[i] > yMax) yMax = yValues[i];
        }

        // Define the axes
        var xAxis = d3.scale.linear()
                .domain([xMin, xMax])
                .range([0, width]);

        var yAxis = d3.scale.linear()
                .domain([0, yMax])
                .range([height, 0]);

        // Set up the data:
        var data = lowerBounds.map(function (d, i) {
            return {'width': upperBounds[i] - lowerBounds[i], 'height': yValues[i], 'offset': lowerBounds[i]};
        });


        // Adds the svg canvas
        var svg = d3.select("#" + appendTo.attr("id"))
                .append("svg")
                .attr("width", width + margin.left + margin.right)
                .attr("height", height + margin.top + margin.bottom)
                .attr("padding", "20px")
                .append("g")
                .attr("transform",
                        "translate(" + margin.left + "," + margin.top + ")");

        // Add the bins.
        svg.selectAll(".bin")
                .data(data)
                .enter().append("rect")
                .attr("class", "bin")
                .attr("x", function(d) { return xAxis(d.offset); })
                .attr("width", function(d) { return xAxis(xMin + d.width) - 1; })
                .attr("y", function(d) { return yAxis(d.height); })
                .attr("height", function(d) { return height - yAxis(d.height); });

        svg.append("g")
                .attr("class", "x axis")
                .attr("transform", "translate(0," + height + ")")
                .call(d3.svg.axis()
                        .scale(xAxis)
                        .orient("bottom"));

        svg.append("g")
                .attr("class", "y axis")
                .call(d3.svg.axis()
                        .scale(yAxis)
                        .orient("left"));

        //Add title (if present)
        if (title) {
            svg.append("text")
                    .attr("x", (width / 2))
                    .attr("y", 0 - ((margin.top - 30) / 2))
                    .attr("text-anchor", "middle")
                    .style("font-size", "13px")
                    .style("text-decoration", "underline")
                    .text(title);
        }
    }
</script>

<table style="width: 100%; padding: 5px" class="hd">
    <tbody>
    <tr>
        <td style="width:15px; height:35px; padding: 4px 15px;">
        <td>Data Analysis</td>
        <td style="text-align:right">${datetime}</td>
        <td style="width:15px; height:35px; padding: 4px 15px;">
    </tr>
    </tbody>
</table>

<div style="width:1400px; margin:0 auto; border:0px" id="outerdiv">
    <div style="width:100%; padding-top:20px" id="maindiv">
        <div style="width:100%; height:20px"></div>
        <div style="width:100%;" class="sectionheader">
            <div style="padding-left:40px; padding-top:3px; padding-bottom:3px">
                Summary Column Analysis
            </div>
        </div>
        <div style="width:100%; height:auto" align="center" id="tablediv">

        </div>

        <div style="width:100%; height:20px"></div>
        <div style="width:100%;" class="sectionheader">
            <div style="padding-left:40px; padding-top:3px; padding-bottom:3px">
                Numerical Column Histograms
            </div>
        </div>
        <div style="width:100%; height:auto" align="center" id="histogramdiv">

        </div>

    </div>
</div>

<#list divs as div>
<div id="${div.id}" style="display:none">
${div.content}
</div>
</#list>

</body>

</html>