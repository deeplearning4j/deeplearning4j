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

class ChartTimeline extends Chart implements Renderable {

    private laneNames:string[];
    private laneData:any[][];

    private lanes:any;
    private itemData:any;
    private mainView:any;
    private miniView:any;
    private brush:any;

    private x:any;
    private x1:any;
    private xTimeAxis:any;
    private y1:any;
    private y2:any;

    private itemRects:any;
    private rect:any;

    private static MINI_LANE_HEIGHT_PX = 12;
    private static ENTRY_LANE_HEIGHT_OFFSET_FRACTION:number = 0.05;
    private static ENTRY_LANE_HEIGHT_TOTAL_FRACTION:number = 0.90;

    private static MILLISEC_PER_MINUTE:number = 60 * 1000;
    private static MILLISEC_PER_HOUR:number = 60 * ChartTimeline.MILLISEC_PER_MINUTE;
    private static MILLISEC_PER_DAY:number = 24 * ChartTimeline.MILLISEC_PER_HOUR;
    private static MILLISEC_PER_WEEK:number = 7 * ChartTimeline.MILLISEC_PER_DAY;

    private static DEFAULT_COLOR = "LightGrey";


    constructor(jsonStr:string) {
        super(ComponentType.ChartTimeline, jsonStr);

        var json = JSON.parse(jsonStr);
        if (!json["componentType"]) json = json[ComponentType[ComponentType.ChartTimeline]];

        this.laneNames = json['laneNames'];
        this.laneData = json['laneData'];
    }


    render = (appendToObject:JQuery) => {
        var instance = this;
        var s:StyleChart = this.getStyle();
        var margin:Margin = Style.getMargins(s);

        //Format data
        this.itemData = [];
        var count = 0;
        for (var i = 0; i < this.laneData.length; i++) {
            for (var j = 0; j < this.laneData[i].length; j++) {
                var obj = {};
                obj["start"] = this.laneData[i][j]["startTimeMs"];
                obj["end"] = this.laneData[i][j]["endTimeMs"];
                obj["id"] = count++;
                obj["lane"] = i;
                obj["color"] = this.laneData[i][j]["color"];
                obj["label"] = this.laneData[i][j]["entryLabel"];
                this.itemData.push(obj);
            }
        }

        this.lanes = [];
        for (var i = 0; i < this.laneNames.length; i++) {
            var obj = {};
            obj["label"] = this.laneNames[i];
            obj["id"] = i;
            this.lanes.push(obj);
        }

        // Adds the svg canvas
        //TODO don't hardcode these colors/attributes...
        var svg = d3.select("#" + appendToObject.attr("id"))
            .append("svg")
            .style("stroke-width", ( s && s.getStrokeWidth() ? s.getStrokeWidth() : ChartConstants.DEFAULT_CHART_STROKE_WIDTH))
            .style("fill", "none")
            .attr("width", s.getWidth())
            .attr("height", s.getHeight())
            .append("g");

        var heightExMargins = s.getHeight() - margin.top - margin.bottom;
        var widthExMargins = s.getWidth() - margin.left - margin.right;
        var miniHeight = this.laneNames.length * ChartTimeline.MINI_LANE_HEIGHT_PX;
        var mainHeight = s.getHeight() - miniHeight - margin.top - margin.bottom - 25;

        var minTime:number = d3.min(this.itemData, function (d:any) { return d.start; });
        var maxTime:number = d3.max(this.itemData, function (d:any) { return d.end; });
        this.x = d3.time.scale()
            .domain([minTime, maxTime])
            .range([0, widthExMargins]);
        this.x1 = d3.time.scale().range([0, widthExMargins]);

        this.y1 = d3.scale.linear().domain([0, this.laneNames.length]).range([0, mainHeight]);
        this.y2 = d3.scale.linear().domain([0, this.laneNames.length]).range([0, miniHeight]);

        //Add a rectangle for clipping the elements in each swimlane
        this.rect = svg.append('defs').append('clipPath')
            .attr('id', 'clip')
            .append('rect')
            .attr('width', widthExMargins)
            .attr('height', s.getHeight() - 100);

        this.mainView = svg.append('g')
            .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')')
            .attr('width', widthExMargins)
            .attr('height', mainHeight)
            .attr('font-size', '12px')
            .attr('font', 'sans-serif');

        this.miniView = svg.append('g')
            .attr('transform', 'translate(' + margin.left + ',' + (mainHeight + margin.top + 25) + ')') //25 being space for ticks/time label
            .attr('width', widthExMargins)
            .attr('height', miniHeight)
            .attr('font-size', '10px')
            .attr('font', 'sans-serif');

        // Horizontal lane divider lines for mainView chart:
        this.mainView.append('g').selectAll('.laneLines')
            .data(this.lanes)
            .enter().append('line')
            .attr('x1', 0)
            .attr('y1', function (d:any) {
                return d3.round(instance.y1(d.id)) + 0.5;
            })
            .attr('x2', widthExMargins)
            .attr('y2', function (d:any) {
                return d3.round(instance.y1(d.id)) + 0.5;
            })
            .attr('stroke', 'lightgray')
            .attr('stroke-width', 1);

        //Add labels for lane text
        this.mainView.append('g').selectAll('.laneText')
            .data(this.lanes)
            .enter().append('text')
            .text(function (d:any) {
                if(d.label) return d.label;
                return "";
            })
            .attr('x', -10)
            .attr('y', function (d:any) {
                return instance.y1(d.id + .5);
            })
            .attr('text-anchor', 'end')
            .attr("font","8pt sans-serif")
            .attr('fill', 'black');

        // Divider lines for miniView chart
        this.miniView.append('g').selectAll('.laneLines')
            .data(this.lanes)
            .enter().append('line')
            .attr('x1', 0)
            .attr('y1', function (d:any) { return d3.round(instance.y2(d.id)) + 0.5; })
            .attr('x2', widthExMargins)
            .attr('y2', function (d:any) { return d3.round(instance.y2(d.id)) + 0.5; })
            .attr('stroke', 'gray')
            .attr('stroke-width', 1.0);

        //Text for mini view
        this.miniView.append('g').selectAll('.laneText')
            .data(this.lanes)
            .enter().append('text')
            .text(function (d:any) {
                if(d.label) return d.label;
                return "";
            })
            .attr('x', -10)
            .attr('y', function (d:any) {
                return instance.y2(d.id + .5);
            })
            .attr('dy', '0.5ex')
            .attr('text-anchor', 'end')
            .attr('fill', 'black');

        // Render time axis
        this.xTimeAxis = d3.svg.axis()
            .scale(this.x1)
            .orient('bottom')
            .ticks(d3.time.days, 1)
            .tickFormat(d3.time.format('%a %d'))
            .tickSize(6, 0);

        //Time axis
        var temp:any = this.mainView.append('g')
            .attr('transform', 'translate(0,' + mainHeight + ')')
            // .attr('class', 'mainView axis time')
            .attr('class', 'timeAxis')
            .attr('fill', 'black')
            .style("stroke", "black").style("stroke-width", 1.0).style("fill", "black")
            .attr("font", "10px sans-serif")
            .call(this.xTimeAxis);
        temp.selectAll('text').style("stroke-width", 0.0).attr('stroke-width', 0.0);

        // draw the itemData
        this.itemRects = this.mainView.append('g')
            .attr('clip-path', 'url(#clip)');

        //Entries for miniView chart
        this.miniView.append('g').selectAll('miniItems')
            .data(this.getMiniViewPaths(this.itemData))
            .enter().append('path')
            .attr('class', function (d:any) {
                return 'miniItem ' + d.class;
            })
            .attr('d', function (d:any) {
                return d.path;
            })
            .attr('stroke', 'black')
            .attr('stroke-width', 'black');

        // Draw the brush selection area (default - set extent to all data)
        this.miniView.append('rect')
            .attr('pointer-events', 'painted')
            .attr('width', widthExMargins)
            .attr('height', miniHeight)
            .attr('visibility', 'hidden')
            .on('mouseup', this.moveBrush);
        this.brush = d3.svg.brush()
            .x(this.x)
            .extent([minTime, maxTime])
            .on("brush", this.renderChart);
        this.miniView.append('g')
            .attr('class', 'x brush')
            .call(this.brush)
            .selectAll('rect')
            .attr('y', 1)
            .attr('height', miniHeight - 1)
            .style('fill','gray')
            .style('fill-opacity','0.2')
            .style('stroke','DarkSlateGray')
            .style('stroke-width',1);


        this.miniView.selectAll('rect.background').remove();
        this.renderChart();

        //Add title (if present)
        if (this.title) {
            var titleStyle:StyleText;
            if (this.style) titleStyle = this.style.getTitleStyle();
            var text = svg.append("text")
                .text(this.title)
                .attr("x", (s.getWidth() / 2))
                .attr("y", ((margin.top - 30) / 2))
                .attr("text-anchor", "middle");

            if (titleStyle) {
                if (titleStyle.getFont()) text.attr("font-family", titleStyle.getFont);
                if (titleStyle.getFontSize() != null) text.attr("font-size", titleStyle.getFontSize() + "pt");
                if (titleStyle.getUnderline() != null) text.style("text-decoration", "underline");
                if (titleStyle.getColor()) text.style("fill", titleStyle.getColor);
                else text.style("fill", ChartConstants.DEFAULT_TITLE_COLOR);
            } else {
                text.style("text-decoration", "underline");
                text.style("fill", ChartConstants.DEFAULT_TITLE_COLOR);
            }
        }
    };


    renderChart = () => {
        var instance:any = this;

        var extent:number[] = this.brush.extent();
        var minExtent:number = extent[0];
        var maxExtent:number = extent[1];

        var visibleItems:any = this.itemData.filter(function (d) {
            return d.start < maxExtent && d.end > minExtent
        });

        this.miniView.select('.brush').call(this.brush.extent([minExtent, maxExtent]));

        this.x1.domain([minExtent, maxExtent]);

        //https://github.com/d3/d3-time-format#timeFormat
        var range = maxExtent - minExtent;
        if (range > 2 * ChartTimeline.MILLISEC_PER_WEEK) {
            this.xTimeAxis.ticks(d3.time.mondays, 1).tickFormat(d3.time.format('%a %d'));
        } else if (range > 2 * ChartTimeline.MILLISEC_PER_DAY) {
            this.xTimeAxis.ticks(d3.time.days, 1).tickFormat(d3.time.format('%a %d'));
        } else if (range > 2 * ChartTimeline.MILLISEC_PER_HOUR) {
            this.xTimeAxis.ticks(d3.time.hours, 4).tickFormat(d3.time.format('%H %p'));
        } else if (range > 2 * ChartTimeline.MILLISEC_PER_MINUTE) {
            this.xTimeAxis.ticks(d3.time.minutes, 1).tickFormat(d3.time.format('%H:%M'));
        } else if (range >= 30000) {
            this.xTimeAxis.ticks(d3.time.seconds, 10).tickFormat(d3.time.format('%H:%M:%S'));
        } else {
            this.xTimeAxis.ticks(d3.time.seconds, 1).tickFormat(d3.time.format('%H:%M:%S'));
        } //no d3.time.milliseconds, so ticks below 1 second are not possible? (or, at least not using same approach as here)

        // Update the axis
        this.mainView.select('.timeAxis').call(this.xTimeAxis);

        // Update the rectangles
        var rects:any = this.itemRects.selectAll('rect')
                .data(visibleItems, function (d) { return d.id; })
                .attr('x', function (d) { return instance.x1(d.start); })
                .attr('width', function (d) { return instance.x1(d.end) - instance.x1(d.start); });

        //Set attributes for mainView swimlane rectangles
        rects.enter().append('rect')
            .attr('x', function (d) { return instance.x1(d.start); })
            .attr('y', function (d) { return instance.y1(d.lane) + ChartTimeline.ENTRY_LANE_HEIGHT_OFFSET_FRACTION * instance.y1(1) + 0.5; })
            .attr('width', function (d) { return instance.x1(d.end) - instance.x1(d.start); })
            .attr('height', function (d) { return ChartTimeline.ENTRY_LANE_HEIGHT_TOTAL_FRACTION * instance.y1(1); })
            .attr('stroke', 'black')
            .attr('fill', function(d){
                if(d.color) return d.color;
                return ChartTimeline.DEFAULT_COLOR;
            })
            .attr('stroke-width', 1);
        rects.exit().remove();

        // Update the item labels
        var labels:any = this.itemRects.selectAll('text')
            .data(visibleItems, function (d) {
                return d.id;
            })
            .attr('x', function (d) {
                return instance.x1(Math.max(d.start, minExtent)) + 2;
            })
            .attr('fill', 'black');

        labels.enter().append('text')
            .text(function (d) {
                if(instance.x1(d.end) - instance.x1(d.start) <= 30) return "";
                if(d.label) return d.label;
                return "";
            })
            .attr('x', function (d) {
                return instance.x1(Math.max(d.start, minExtent)) + 2;
            })
            .attr('y', function (d) {
                return instance.y1(d.lane) + .4 * instance.y1(1) + 0.5;
            })
            .attr('text-anchor', 'start')
            .attr('class', 'itemLabel')
            .attr('fill', 'black');

        labels.exit().remove();
    };

    moveBrush = () => {
        var origin:any = d3.mouse(this.rect[0]);
        var time: any = this.x.invert(origin[0]).getTime();
        var halfExtent: number = (this.brush.extent()[1].getTime() - this.brush.extent()[0].getTime()) / 2;

        this.brush.extent([new Date(time - halfExtent), new Date(time + halfExtent)]);
        this.renderChart();
    };

    getMiniViewPaths = (items:any) => {
        var paths = {}, d, offset = .5 * this.y2(1) + 0.5, result = [];
        for (var i = 0; i < items.length; i++) {
            d = items[i];
            if (!paths[d.class]) paths[d.class] = '';
            paths[d.class] += ['M', this.x(d.start), (this.y2(d.lane) + offset), 'H', this.x(d.end)].join(' ');
        }

        for (var className in paths) {
            result.push({class: className, path: paths[className]});
        }
        return result;
    }
}