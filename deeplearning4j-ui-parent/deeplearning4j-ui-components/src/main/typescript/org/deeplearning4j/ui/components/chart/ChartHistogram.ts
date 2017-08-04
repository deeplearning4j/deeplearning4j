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

class ChartHistogram extends Chart implements Renderable {

    private lowerBounds: number[];
    private upperBounds: number[];
    private yValues: number[];

    constructor(jsonStr: string){
        super(ComponentType.ChartHistogram, jsonStr);

        var json = JSON.parse(jsonStr);
        if(!json["componentType"]) json = json[ComponentType[ComponentType.ChartHistogram]];


        this.lowerBounds = json['lowerBounds'];
        this.upperBounds = json['upperBounds'];
        this.yValues = json['yvalues'];
    }


    render = (appendToObject: JQuery) => {
        var s: StyleChart = this.getStyle();
        var margin: Margin = Style.getMargins(s);

        // Add the bins.
        var xMin: number;
        var xMax: number;
        var yMin: number;
        var yMax: number;
        if(this.setXMin) xMin = this.setXMin;
        else xMin = (this.lowerBounds ? d3.min(this.lowerBounds) : 0);
        if(this.setXMax) xMax = this.setXMax;
        else xMax = (this.upperBounds ? d3.max(this.upperBounds) : 1);
        if(this.setYMin) yMin = this.setYMin;
        else yMin = 0;
        if(this.setYMax) yMax = this.setYMax;
        else yMax = (this.yValues ? d3.max(this.yValues) : 1);

        //// Define the axes
        var xScale: any = d3.scale.linear()
            .domain([xMin, xMax])
            .range([0, margin.widthExMargins]);

        var xAxis: any = d3.svg.axis().scale(xScale)
            .orient("bottom").ticks(5);

        if(this.gridVerticalStrokeWidth && this.gridVerticalStrokeWidth > 0){
            xAxis.innerTickSize(-margin.heightExMargins);     //used as grid line
        }

        var yScale: any = d3.scale.linear()
            .domain([0, yMax])
            .range([margin.heightExMargins, 0]);
        var yAxis: any = d3.svg.axis().scale(yScale)
            .orient("left").ticks(5);
        if(this.gridHorizontalStrokeWidth && this.gridHorizontalStrokeWidth > 0){
            yAxis.innerTickSize(-margin.widthExMargins);      //used as grid line
        }



        if(this.suppressAxisHorizontal === true) xAxis.tickValues([]);

        if(this.suppressAxisVertical === true) yAxis.tickValues([]);

        // Set up the data:
        var lowerBounds: number[] = this.lowerBounds;
        var upperBounds: number[] = this.upperBounds;
        var yValues: number[] = this.yValues;

        var data: any = lowerBounds.map(function (d, i) {
            return {'width': upperBounds[i] - lowerBounds[i], 'height': yValues[i], 'offset': lowerBounds[i]};
        });

        // Adds the svg canvas
        var svg = d3.select("#" + appendToObject.attr("id"))
            .append("svg")
            .style("fill", "none")
            .attr("width", s.getWidth())
            .attr("height", s.getHeight())
            .attr("padding", "20px")
            .append("g")
            .attr("transform",
                "translate(" + margin.left + "," + margin.top + ")");



        svg.selectAll(".bin")
            .data(data)
            .enter().append("rect")
            .attr("class", "bin")
            .style("fill","steelblue")
            .attr("x", function(d: any) { return xScale(d.offset); })
            .attr("width", function(d: any) { return xScale(xMin + d.width) - 1; })
            .attr("y", function(d: any) { return yScale(d.height); })
            .attr("height", function(d: any) { return margin.heightExMargins - yScale(d.height); });

        // Add the X Axis
        var xAxisNode = svg.append("g")
            .attr("class", "x axis")
            .attr("transform", "translate(0," + margin.heightExMargins + ")")
            .style("stroke","#000")
            .style("stroke-width", (s != null && s.getAxisStrokeWidth() != null ? s.getAxisStrokeWidth() : ChartConstants.DEFAULT_AXIS_STROKE_WIDTH))
            .style("fill","none")
            .call(xAxis);
        xAxisNode.selectAll('text').style("stroke-width",0).style("fill","#000000");

        if(this.gridVerticalStrokeWidth != null) xAxisNode.selectAll('.axis line').style({'stroke-width': this.gridVerticalStrokeWidth});

        // Add the Y Axis
        var yAxisNode = svg.append("g")
            .attr("class", "y axis")
            .style("stroke","#000")
            .style("stroke-width", (s != null && s.getAxisStrokeWidth() != null ? s.getAxisStrokeWidth() : ChartConstants.DEFAULT_AXIS_STROKE_WIDTH))
            .style("fill","none")
            .call(yAxis);
        yAxisNode.selectAll('text').style("stroke-width",0).style("fill","#000000");

        if(this.gridHorizontalStrokeWidth != null) yAxisNode.selectAll('.axis line').style({'stroke-width': this.gridHorizontalStrokeWidth});

        //Add title (if present)
        if (this.title) {
            var titleStyle: StyleText;
            if(this.style) titleStyle = this.style.getTitleStyle();
            Chart.appendTitle(svg, this.title, margin, titleStyle);
        }
    }
}