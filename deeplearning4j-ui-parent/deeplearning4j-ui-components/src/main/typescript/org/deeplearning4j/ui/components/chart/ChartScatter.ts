/*
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */
/// <reference path="../../api/Component.ts" />
/// <reference path="../../typedefs/d3.d.ts" />
/// <reference path="../../util/TSUtils.ts" />
/// <reference path="Chart.ts" />

class ChartScatter extends Chart implements Renderable {

    private xData:number[][];
    private yData:number[][];
    private seriesNames:string[];

    constructor(jsonStr:string) {
        super(ComponentType.ChartScatter, jsonStr);

        var json = JSON.parse(jsonStr);
        if(!json["componentType"]) json = json[ComponentType[ComponentType.ChartScatter]];

        this.xData = json['x'];
        this.yData = json['y'];
        this.seriesNames = json['seriesNames'];
    }


    render = (appendToObject:JQuery) => {

        var nSeries:number = (!this.xData ? 0 : this.xData.length);
        var s:StyleChart = this.getStyle();
        var margin:Margin = Style.getMargins(s);

        // Set the ranges
        var xScale:d3.scale.Linear<number,number> = d3.scale.linear().range([0, margin.widthExMargins]);
        var yScale:d3.scale.Linear<number,number> = d3.scale.linear().range([margin.heightExMargins, 0]);

        // Define the axes
        var xAxis = d3.svg.axis().scale(xScale)
            .innerTickSize(-margin.heightExMargins)     //used as grid line
            .orient("bottom").ticks(5);
        var yAxis = d3.svg.axis().scale(yScale)
            .innerTickSize(-margin.widthExMargins)      //used as grid line
            .orient("left").ticks(5);

        if (this.suppressAxisHorizontal === true) xAxis.tickValues([]);

        if (this.suppressAxisVertical === true) yAxis.tickValues([]);

        // Adds the svg canvas
        //TODO don't hardcode these colors/attributes...
        var svg = d3.select("#" + appendToObject.attr("id"))
            .append("svg")
            .style("stroke-width", ( s && s.getStrokeWidth() ? s.getStrokeWidth() : 1))
            .style("fill", "none")
            .attr("width", s.getWidth())
            .attr("height", s.getHeight())
            .attr("padding", "20px")
            .append("g")
            .attr("transform",
                "translate(" + margin.left + "," + margin.top + ")");

        // Scale the range of the chart
        var xMin:number;
        var xMax:number;
        var yMin:number;
        var yMax:number;
        if (this.setXMin) xMin = this.setXMin;
        else xMin = (this.xData ? TSUtils.min(this.xData) : 0);
        if (this.setXMax) xMax = this.setXMax;
        else xMax = (this.xData ? TSUtils.max(this.xData) : 1);
        if (this.setYMin) yMin = this.setYMin;
        else yMin = (this.yData ? TSUtils.min(this.yData) : 0);
        if (this.setYMax) yMax = this.setYMax;
        else yMax = (this.yData ? TSUtils.max(this.yData) : 1);

        xScale.domain([xMin, xMax]);
        yScale.domain([yMin, yMax]);

        // Add the valueline path.
        var defaultColor:Ordinal<string,string> = d3.scale.category10();
        for (var i = 0; i < nSeries; i++) {
            var xVals = this.xData[i];
            var yVals = this.yData[i];

            var data = xVals.map(function (d, i) {
                return {'xPos': xVals[i], 'yPos': yVals[i]};
            });

            svg.selectAll("circle")
                .data(data)
                .enter()
                .append("circle")
                .style("fill", (s && s.getSeriesColor(i) ? s.getSeriesColor(i) : defaultColor(String(i))))
                .attr("r", (s && s.getPointSize() ? s.getPointSize() : ChartConstants.DEFAULT_CHART_POINT_SIZE))
                .attr("cx", function (d) {
                    return xScale(d['xPos']);
                })
                .attr("cy", function (d) {
                    return yScale(d['yPos']);
                });
        }

        // Add the X Axis
        var xAxisNode = svg.append("g")
            .attr("class", "x axis")
            .attr("transform", "translate(0," + margin.heightExMargins + ")")
            .style("stroke", "#000")
            .style("stroke-width", (s != null && s.getAxisStrokeWidth() != null ? s.getAxisStrokeWidth() : ChartConstants.DEFAULT_AXIS_STROKE_WIDTH))
            .style("fill", "none")
            .call(xAxis);
        xAxisNode.selectAll('text').style("stroke-width",0).style("fill","#000000");

        if (this.gridVerticalStrokeWidth != null) xAxisNode.selectAll('.axis line').style({'stroke-width': this.gridVerticalStrokeWidth});

        // Add the Y Axis
        var yAxisNode = svg.append("g")
            .attr("class", "y axis")
            .style("stroke", "#000")
            .style("stroke-width", (s != null && s.getAxisStrokeWidth() != null ? s.getAxisStrokeWidth() : ChartConstants.DEFAULT_AXIS_STROKE_WIDTH))
            .style("fill", "none")
            .call(yAxis);
        yAxisNode.selectAll('text').style("stroke-width",0).style("fill","#000000");

        if (this.gridHorizontalStrokeWidth != null) yAxisNode.selectAll('.axis line').style({'stroke-width': this.gridHorizontalStrokeWidth});

        //Add legend (if present)
        if (this.seriesNames && this.showLegend === true) {
            var legendSpace = margin.widthExMargins / i;
            for (var i = 0; i < nSeries; i++) {
                var values = this.xData[i];
                var yValues = this.yData[i];
                var lastX = values[values.length - 1];
                var lastY = yValues[yValues.length - 1];
                var toDisplay;
                if (!lastX || !lastY) toDisplay = this.seriesNames[i] + " (no data)";
                else toDisplay = this.seriesNames[i] + " (" + lastX.toPrecision(5) + "," + lastY.toPrecision(5) + ")";
                svg.append("text")
                    .attr("x", (legendSpace / 2) + i * legendSpace) // spacing
                    .attr("y", margin.heightExMargins + (margin.bottom / 2) + 5)
                    .attr("class", "legend")    // style the legend
                    .style("fill", (s && s.getSeriesColor(i) ? s.getSeriesColor(i) : defaultColor(String(i))))
                    .text(toDisplay);
            }
        }

        //Add title (if present)
        if (this.title) {
            var titleStyle: StyleText;
            if(this.style) titleStyle = this.style.getTitleStyle();
            Chart.appendTitle(svg, this.title, margin, titleStyle);
        }
    }
}