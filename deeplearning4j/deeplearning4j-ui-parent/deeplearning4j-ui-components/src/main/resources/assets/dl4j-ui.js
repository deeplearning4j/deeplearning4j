/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var Style = (function () {
    function Style(jsonObj) {
        var _this = this;
        this.getWidth = function () { return _this.width; };
        this.getHeight = function () { return _this.height; };
        this.getWidthUnit = function () { return _this.widthUnit; };
        this.getHeightUnit = function () { return _this.heightUnit; };
        this.getMarginTop = function () { return _this.marginTop; };
        this.getMarginBottom = function () { return _this.marginBottom; };
        this.getMarginLeft = function () { return _this.marginLeft; };
        this.getMarginRight = function () { return _this.marginRight; };
        this.getBackgroundColor = function () { return _this.backgroundColor; };
        this.width = jsonObj['width'];
        this.height = jsonObj['height'];
        this.widthUnit = TSUtils.normalizeLengthUnit(jsonObj['widthUnit']);
        this.heightUnit = TSUtils.normalizeLengthUnit(jsonObj['heightUnit']);
        this.marginTop = jsonObj['marginTop'];
        this.marginBottom = jsonObj['marginBottom'];
        this.marginLeft = jsonObj['marginLeft'];
        this.marginRight = jsonObj['marginRight'];
        this.backgroundColor = jsonObj['backgroundColor'];
    }
    Style.getMargins = function (s) {
        var mTop = (s ? s.getMarginTop() : 0);
        var mBottom = (s ? s.getMarginBottom() : 0);
        var mLeft = (s ? s.getMarginLeft() : 0);
        var mRight = (s ? s.getMarginRight() : 0);
        return { top: mTop,
            right: mRight,
            bottom: mBottom,
            left: mLeft,
            widthExMargins: s.getWidth() - mLeft - mRight,
            heightExMargins: s.getHeight() - mTop - mBottom };
    };
    return Style;
}());
var ComponentType;
(function (ComponentType) {
    ComponentType[ComponentType["ComponentText"] = 0] = "ComponentText";
    ComponentType[ComponentType["ComponentTable"] = 1] = "ComponentTable";
    ComponentType[ComponentType["ComponentDiv"] = 2] = "ComponentDiv";
    ComponentType[ComponentType["ChartHistogram"] = 3] = "ChartHistogram";
    ComponentType[ComponentType["ChartHorizontalBar"] = 4] = "ChartHorizontalBar";
    ComponentType[ComponentType["ChartLine"] = 5] = "ChartLine";
    ComponentType[ComponentType["ChartScatter"] = 6] = "ChartScatter";
    ComponentType[ComponentType["ChartStackedArea"] = 7] = "ChartStackedArea";
    ComponentType[ComponentType["ChartTimeline"] = 8] = "ChartTimeline";
    ComponentType[ComponentType["DecoratorAccordion"] = 9] = "DecoratorAccordion";
})(ComponentType || (ComponentType = {}));
var Component = (function () {
    function Component(componentType) {
        this.componentType = componentType;
    }
    Component.prototype.getComponentType = function () {
        return this.componentType;
    };
    Component.getComponent = function (jsonStr) {
        var json = JSON.parse(jsonStr);
        var key;
        if (json["componentType"])
            key = json["componentType"];
        else
            key = Object.keys(json)[0];
        switch (key) {
            case ComponentType[ComponentType.ComponentText]:
                return new ComponentText(jsonStr);
            case ComponentType[ComponentType.ComponentTable]:
                return new ComponentTable(jsonStr);
            case ComponentType[ComponentType.ChartHistogram]:
                return new ChartHistogram(jsonStr);
            case ComponentType[ComponentType.ChartHorizontalBar]:
                throw new Error("Horizontal bar chart: not yet implemented");
            case ComponentType[ComponentType.ChartLine]:
                return new ChartLine(jsonStr);
            case ComponentType[ComponentType.ChartScatter]:
                return new ChartScatter(jsonStr);
            case ComponentType[ComponentType.ChartStackedArea]:
                return new ChartStackedArea(jsonStr);
            case ComponentType[ComponentType.ChartTimeline]:
                return new ChartTimeline(jsonStr);
            case ComponentType[ComponentType.DecoratorAccordion]:
                return new DecoratorAccordion(jsonStr);
            case ComponentType[ComponentType.ComponentDiv]:
                return new ComponentDiv(jsonStr);
            default:
                throw new Error("Unknown component type \"" + key + "\" or invalid JSON: \"" + jsonStr + "\"");
        }
    };
    return Component;
}());
var ChartConstants = (function () {
    function ChartConstants() {
    }
    ChartConstants.DEFAULT_CHART_STROKE_WIDTH = 1.0;
    ChartConstants.DEFAULT_CHART_POINT_SIZE = 3.0;
    ChartConstants.DEFAULT_AXIS_STROKE_WIDTH = 1.0;
    ChartConstants.DEFAULT_TITLE_COLOR = "#000000";
    return ChartConstants;
}());
var TSUtils = (function () {
    function TSUtils() {
    }
    TSUtils.max = function (input) {
        var max = -Number.MAX_VALUE;
        for (var i = 0; i < input.length; i++) {
            for (var j = 0; j < input[i].length; j++) {
                max = Math.max(max, input[i][j]);
            }
        }
        return max;
    };
    TSUtils.min = function (input) {
        var min = Number.MAX_VALUE;
        for (var i = 0; i < input.length; i++) {
            for (var j = 0; j < input[i].length; j++) {
                min = Math.min(min, input[i][j]);
            }
        }
        return min;
    };
    TSUtils.normalizeLengthUnit = function (input) {
        if (input == null)
            return input;
        switch (input.toLowerCase()) {
            case "px":
                return "px";
            case "percent":
            case "%":
                return "%";
            case "cm":
                return "cm";
            case "mm":
                return "mm";
            case "in":
                return "in";
            default:
                return input;
        }
    };
    return TSUtils;
}());
var Chart = (function (_super) {
    __extends(Chart, _super);
    function Chart(componentType, jsonStr) {
        var _this = _super.call(this, componentType) || this;
        var jsonOrig = JSON.parse(jsonStr);
        var json = JSON.parse(jsonStr);
        if (!json["componentType"])
            json = json[ComponentType[componentType]];
        _this.suppressAxisHorizontal = json['suppressAxisHorizontal'];
        _this.suppressAxisVertical = json['suppressAxisVertical'];
        _this.showLegend = json['showLegend'];
        _this.title = json['title'];
        _this.setXMin = json['setXMin'];
        _this.setXMax = json['setXMax'];
        _this.setYMin = json['setYMin'];
        _this.setYMax = json['setYMax'];
        _this.gridVerticalStrokeWidth = json['gridVerticalStrokeWidth'];
        _this.gridHorizontalStrokeWidth = json['gridHorizontalStrokeWidth'];
        if (json['style'])
            _this.style = new StyleChart(json['style']);
        return _this;
    }
    Chart.prototype.getStyle = function () {
        return this.style;
    };
    Chart.appendTitle = function (svg, title, margin, titleStyle) {
        var text = svg.append("text")
            .text(title)
            .attr("x", (margin.widthExMargins / 2))
            .attr("y", 0 - ((margin.top - 30) / 2))
            .attr("text-anchor", "middle");
        if (titleStyle) {
            if (titleStyle.getFont())
                text.attr("font-family", titleStyle.getFont);
            if (titleStyle.getFontSize() != null)
                text.attr("font-size", titleStyle.getFontSize() + "pt");
            if (titleStyle.getUnderline() != null)
                text.style("text-decoration", "underline");
            if (titleStyle.getColor())
                text.style("fill", titleStyle.getColor);
            else
                text.style("fill", ChartConstants.DEFAULT_TITLE_COLOR);
        }
        else {
            text.style("text-decoration", "underline");
            text.style("fill", ChartConstants.DEFAULT_TITLE_COLOR);
        }
    };
    return Chart;
}(Component));
var ChartHistogram = (function (_super) {
    __extends(ChartHistogram, _super);
    function ChartHistogram(jsonStr) {
        var _this = _super.call(this, ComponentType.ChartHistogram, jsonStr) || this;
        _this.render = function (appendToObject) {
            var s = _this.getStyle();
            var margin = Style.getMargins(s);
            var xMin;
            var xMax;
            var yMin;
            var yMax;
            if (_this.setXMin)
                xMin = _this.setXMin;
            else
                xMin = (_this.lowerBounds ? d3.min(_this.lowerBounds) : 0);
            if (_this.setXMax)
                xMax = _this.setXMax;
            else
                xMax = (_this.upperBounds ? d3.max(_this.upperBounds) : 1);
            if (_this.setYMin)
                yMin = _this.setYMin;
            else
                yMin = 0;
            if (_this.setYMax)
                yMax = _this.setYMax;
            else
                yMax = (_this.yValues ? d3.max(_this.yValues) : 1);
            var xScale = d3.scale.linear()
                .domain([xMin, xMax])
                .range([0, margin.widthExMargins]);
            var xAxis = d3.svg.axis().scale(xScale)
                .orient("bottom").ticks(5);
            if (_this.gridVerticalStrokeWidth && _this.gridVerticalStrokeWidth > 0) {
                xAxis.innerTickSize(-margin.heightExMargins);
            }
            var yScale = d3.scale.linear()
                .domain([0, yMax])
                .range([margin.heightExMargins, 0]);
            var yAxis = d3.svg.axis().scale(yScale)
                .orient("left").ticks(5);
            if (_this.gridHorizontalStrokeWidth && _this.gridHorizontalStrokeWidth > 0) {
                yAxis.innerTickSize(-margin.widthExMargins);
            }
            if (_this.suppressAxisHorizontal === true)
                xAxis.tickValues([]);
            if (_this.suppressAxisVertical === true)
                yAxis.tickValues([]);
            var lowerBounds = _this.lowerBounds;
            var upperBounds = _this.upperBounds;
            var yValues = _this.yValues;
            var data = lowerBounds.map(function (d, i) {
                return { 'width': upperBounds[i] - lowerBounds[i], 'height': yValues[i], 'offset': lowerBounds[i] };
            });
            var svg = d3.select("#" + appendToObject.attr("id"))
                .append("svg")
                .style("fill", "none")
                .attr("width", s.getWidth())
                .attr("height", s.getHeight())
                .attr("padding", "20px")
                .append("g")
                .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
            svg.selectAll(".bin")
                .data(data)
                .enter().append("rect")
                .attr("class", "bin")
                .style("fill", "steelblue")
                .attr("x", function (d) { return xScale(d.offset); })
                .attr("width", function (d) { return xScale(xMin + d.width) - 1; })
                .attr("y", function (d) { return yScale(d.height); })
                .attr("height", function (d) { return margin.heightExMargins - yScale(d.height); });
            var xAxisNode = svg.append("g")
                .attr("class", "x axis")
                .attr("transform", "translate(0," + margin.heightExMargins + ")")
                .style("stroke", "#000")
                .style("stroke-width", (s != null && s.getAxisStrokeWidth() != null ? s.getAxisStrokeWidth() : ChartConstants.DEFAULT_AXIS_STROKE_WIDTH))
                .style("fill", "none")
                .call(xAxis);
            xAxisNode.selectAll('text').style("stroke-width", 0).style("fill", "#000000");
            if (_this.gridVerticalStrokeWidth != null)
                xAxisNode.selectAll('.axis line').style({ 'stroke-width': _this.gridVerticalStrokeWidth });
            var yAxisNode = svg.append("g")
                .attr("class", "y axis")
                .style("stroke", "#000")
                .style("stroke-width", (s != null && s.getAxisStrokeWidth() != null ? s.getAxisStrokeWidth() : ChartConstants.DEFAULT_AXIS_STROKE_WIDTH))
                .style("fill", "none")
                .call(yAxis);
            yAxisNode.selectAll('text').style("stroke-width", 0).style("fill", "#000000");
            if (_this.gridHorizontalStrokeWidth != null)
                yAxisNode.selectAll('.axis line').style({ 'stroke-width': _this.gridHorizontalStrokeWidth });
            if (_this.title) {
                var titleStyle;
                if (_this.style)
                    titleStyle = _this.style.getTitleStyle();
                Chart.appendTitle(svg, _this.title, margin, titleStyle);
            }
        };
        var json = JSON.parse(jsonStr);
        if (!json["componentType"])
            json = json[ComponentType[ComponentType.ChartHistogram]];
        _this.lowerBounds = json['lowerBounds'];
        _this.upperBounds = json['upperBounds'];
        _this.yValues = json['yvalues'];
        return _this;
    }
    return ChartHistogram;
}(Chart));
var ChartLine = (function (_super) {
    __extends(ChartLine, _super);
    function ChartLine(jsonStr) {
        var _this = _super.call(this, ComponentType.ChartLine, jsonStr) || this;
        _this.render = function (appendToObject) {
            var nSeries = (!_this.xData ? 0 : _this.xData.length);
            var s = _this.getStyle();
            var margin = Style.getMargins(s);
            var xScale = d3.scale.linear().range([0, margin.widthExMargins]);
            var yScale = d3.scale.linear().range([margin.heightExMargins, 0]);
            var xAxis = d3.svg.axis().scale(xScale)
                .orient("bottom").ticks(5);
            if (_this.gridVerticalStrokeWidth != null && _this.gridVerticalStrokeWidth > 0) {
                xAxis.innerTickSize(-margin.heightExMargins);
            }
            var yAxis = d3.svg.axis().scale(yScale)
                .orient("left").ticks(5);
            if (_this.gridHorizontalStrokeWidth != null && _this.gridHorizontalStrokeWidth > 0) {
                yAxis.innerTickSize(-margin.widthExMargins);
            }
            if (_this.suppressAxisHorizontal === true)
                xAxis.tickValues([]);
            if (_this.suppressAxisVertical === true)
                yAxis.tickValues([]);
            var valueline = d3.svg.line()
                .x(function (d) {
                return xScale(d.xPos);
            })
                .y(function (d) {
                return yScale(d.yPos);
            });
            var svg = d3.select("#" + appendToObject.attr("id"))
                .append("svg")
                .style("stroke-width", (s && s.getStrokeWidth() ? s.getStrokeWidth() : ChartConstants.DEFAULT_CHART_STROKE_WIDTH))
                .style("fill", "none")
                .attr("width", s.getWidth())
                .attr("height", s.getHeight())
                .append("g")
                .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
            var xMin;
            var xMax;
            var yMin;
            var yMax;
            if (_this.setXMin != null)
                xMin = _this.setXMin;
            else
                xMin = (_this.xData ? TSUtils.min(_this.xData) : 0);
            if (_this.setXMax != null)
                xMax = _this.setXMax;
            else
                xMax = (_this.xData ? TSUtils.max(_this.xData) : 1);
            if (_this.setYMin != null)
                yMin = _this.setYMin;
            else
                yMin = (_this.yData ? TSUtils.min(_this.yData) : 0);
            if (_this.setYMax != null)
                yMax = _this.setYMax;
            else
                yMax = (_this.yData ? TSUtils.max(_this.yData) : 1);
            xScale.domain([xMin, xMax]);
            yScale.domain([yMin, yMax]);
            var defaultColor = d3.scale.category10();
            for (var i = 0; i < nSeries; i++) {
                var xVals = _this.xData[i];
                var yVals = _this.yData[i];
                var data = xVals.map(function (d, i) {
                    return { 'xPos': xVals[i], 'yPos': yVals[i] };
                });
                svg.append("path")
                    .attr("class", "line")
                    .style("stroke", (s && s.getSeriesColor(i) ? s.getSeriesColor(i) : defaultColor(String(i))))
                    .attr("d", valueline(data));
            }
            var xAxisNode = svg.append("g")
                .attr("class", "x axis")
                .attr("transform", "translate(0," + margin.heightExMargins + ")")
                .style("stroke", "#000")
                .style("stroke-width", (s != null && s.getAxisStrokeWidth() != null ? s.getAxisStrokeWidth() : ChartConstants.DEFAULT_AXIS_STROKE_WIDTH))
                .style("fill", "none")
                .call(xAxis);
            xAxisNode.selectAll('text').style("stroke-width", 0).style("fill", "#000000");
            if (_this.gridVerticalStrokeWidth != null)
                xAxisNode.selectAll('.axis line').style({ 'stroke-width': _this.gridVerticalStrokeWidth });
            var yAxisNode = svg.append("g")
                .attr("class", "y axis")
                .style("stroke", "#000")
                .style("stroke-width", (s != null && s.getAxisStrokeWidth() != null ? s.getAxisStrokeWidth() : ChartConstants.DEFAULT_AXIS_STROKE_WIDTH))
                .style("fill", "none")
                .call(yAxis);
            yAxisNode.selectAll('text').style("stroke-width", 0).style("fill", "#000000");
            if (_this.gridHorizontalStrokeWidth != null)
                yAxisNode.selectAll('.axis line').style({ 'stroke-width': _this.gridHorizontalStrokeWidth });
            if (_this.seriesNames && _this.showLegend === true) {
                var legendSpace = margin.widthExMargins / i;
                for (var i = 0; i < nSeries; i++) {
                    var values = _this.xData[i];
                    var yValues = _this.yData[i];
                    var lastX = values[values.length - 1];
                    var lastY = yValues[yValues.length - 1];
                    var toDisplay = _this.seriesNames[i];
                    svg.append("text")
                        .attr("x", (legendSpace / 2) + i * legendSpace)
                        .attr("y", margin.heightExMargins + (margin.bottom / 2) + 5)
                        .attr("class", "legend")
                        .style("fill", (s && s.getSeriesColor(i) ? s.getSeriesColor(i) : defaultColor(String(i))))
                        .text(toDisplay);
                }
            }
            if (_this.title) {
                var titleStyle;
                if (_this.style)
                    titleStyle = _this.style.getTitleStyle();
                Chart.appendTitle(svg, _this.title, margin, titleStyle);
            }
        };
        var json = JSON.parse(jsonStr);
        if (!json["componentType"])
            json = json[ComponentType[ComponentType.ChartLine]];
        _this.xData = json['x'];
        _this.yData = json['y'];
        _this.seriesNames = json['seriesNames'];
        return _this;
    }
    return ChartLine;
}(Chart));
var ChartScatter = (function (_super) {
    __extends(ChartScatter, _super);
    function ChartScatter(jsonStr) {
        var _this = _super.call(this, ComponentType.ChartScatter, jsonStr) || this;
        _this.render = function (appendToObject) {
            var nSeries = (!_this.xData ? 0 : _this.xData.length);
            var s = _this.getStyle();
            var margin = Style.getMargins(s);
            var xScale = d3.scale.linear().range([0, margin.widthExMargins]);
            var yScale = d3.scale.linear().range([margin.heightExMargins, 0]);
            var xAxis = d3.svg.axis().scale(xScale)
                .innerTickSize(-margin.heightExMargins)
                .orient("bottom").ticks(5);
            var yAxis = d3.svg.axis().scale(yScale)
                .innerTickSize(-margin.widthExMargins)
                .orient("left").ticks(5);
            if (_this.suppressAxisHorizontal === true)
                xAxis.tickValues([]);
            if (_this.suppressAxisVertical === true)
                yAxis.tickValues([]);
            var svg = d3.select("#" + appendToObject.attr("id"))
                .append("svg")
                .style("stroke-width", (s && s.getStrokeWidth() ? s.getStrokeWidth() : 1))
                .style("fill", "none")
                .attr("width", s.getWidth())
                .attr("height", s.getHeight())
                .attr("padding", "20px")
                .append("g")
                .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
            var xMin;
            var xMax;
            var yMin;
            var yMax;
            if (_this.setXMin)
                xMin = _this.setXMin;
            else
                xMin = (_this.xData ? TSUtils.min(_this.xData) : 0);
            if (_this.setXMax)
                xMax = _this.setXMax;
            else
                xMax = (_this.xData ? TSUtils.max(_this.xData) : 1);
            if (_this.setYMin)
                yMin = _this.setYMin;
            else
                yMin = (_this.yData ? TSUtils.min(_this.yData) : 0);
            if (_this.setYMax)
                yMax = _this.setYMax;
            else
                yMax = (_this.yData ? TSUtils.max(_this.yData) : 1);
            xScale.domain([xMin, xMax]);
            yScale.domain([yMin, yMax]);
            var defaultColor = d3.scale.category10();
            for (var i = 0; i < nSeries; i++) {
                var xVals = _this.xData[i];
                var yVals = _this.yData[i];
                var data = xVals.map(function (d, i) {
                    return { 'xPos': xVals[i], 'yPos': yVals[i] };
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
            var xAxisNode = svg.append("g")
                .attr("class", "x axis")
                .attr("transform", "translate(0," + margin.heightExMargins + ")")
                .style("stroke", "#000")
                .style("stroke-width", (s != null && s.getAxisStrokeWidth() != null ? s.getAxisStrokeWidth() : ChartConstants.DEFAULT_AXIS_STROKE_WIDTH))
                .style("fill", "none")
                .call(xAxis);
            xAxisNode.selectAll('text').style("stroke-width", 0).style("fill", "#000000");
            if (_this.gridVerticalStrokeWidth != null)
                xAxisNode.selectAll('.axis line').style({ 'stroke-width': _this.gridVerticalStrokeWidth });
            var yAxisNode = svg.append("g")
                .attr("class", "y axis")
                .style("stroke", "#000")
                .style("stroke-width", (s != null && s.getAxisStrokeWidth() != null ? s.getAxisStrokeWidth() : ChartConstants.DEFAULT_AXIS_STROKE_WIDTH))
                .style("fill", "none")
                .call(yAxis);
            yAxisNode.selectAll('text').style("stroke-width", 0).style("fill", "#000000");
            if (_this.gridHorizontalStrokeWidth != null)
                yAxisNode.selectAll('.axis line').style({ 'stroke-width': _this.gridHorizontalStrokeWidth });
            if (_this.seriesNames && _this.showLegend === true) {
                var legendSpace = margin.widthExMargins / i;
                for (var i = 0; i < nSeries; i++) {
                    var values = _this.xData[i];
                    var yValues = _this.yData[i];
                    var lastX = values[values.length - 1];
                    var lastY = yValues[yValues.length - 1];
                    var toDisplay;
                    if (!lastX || !lastY)
                        toDisplay = _this.seriesNames[i] + " (no data)";
                    else
                        toDisplay = _this.seriesNames[i] + " (" + lastX.toPrecision(5) + "," + lastY.toPrecision(5) + ")";
                    svg.append("text")
                        .attr("x", (legendSpace / 2) + i * legendSpace)
                        .attr("y", margin.heightExMargins + (margin.bottom / 2) + 5)
                        .attr("class", "legend")
                        .style("fill", (s && s.getSeriesColor(i) ? s.getSeriesColor(i) : defaultColor(String(i))))
                        .text(toDisplay);
                }
            }
            if (_this.title) {
                var titleStyle;
                if (_this.style)
                    titleStyle = _this.style.getTitleStyle();
                Chart.appendTitle(svg, _this.title, margin, titleStyle);
            }
        };
        var json = JSON.parse(jsonStr);
        if (!json["componentType"])
            json = json[ComponentType[ComponentType.ChartScatter]];
        _this.xData = json['x'];
        _this.yData = json['y'];
        _this.seriesNames = json['seriesNames'];
        return _this;
    }
    return ChartScatter;
}(Chart));
var Legend = (function () {
    function Legend() {
    }
    Legend.offsetX = 15;
    Legend.offsetY = 15;
    Legend.padding = 8;
    Legend.separation = 12;
    Legend.boxSize = 10;
    Legend.fillColor = "#FFFFFF";
    Legend.legendOpacity = 0.75;
    Legend.borderStrokeColor = "#000000";
    Legend.legendFn = (function (g) {
        var svg = d3.select(g.property("nearestViewportElement"));
        var legendBox = g.selectAll(".outerRect").data([true]);
        var legendItems = g.selectAll(".legendElement").data([true]);
        legendBox.enter().append("rect").attr("class", "outerRect");
        legendItems.enter().append("g").attr("class", "legendElement");
        var legendElements = [];
        svg.selectAll("[data-legend]").each(function () {
            var thisVar = d3.select(this);
            legendElements.push({
                label: thisVar.attr("data-legend"),
                color: thisVar.style("fill")
            });
        });
        legendItems.selectAll("rect")
            .data(legendElements, function (d) { return d.label; })
            .call(function (d) { d.enter().append("rect"); })
            .call(function (d) { d.exit().remove(); })
            .attr("x", 0)
            .attr("y", function (d, i) { return i * Legend.separation - Legend.boxSize + "px"; })
            .attr("width", Legend.boxSize)
            .attr("height", Legend.boxSize)
            .style("fill", function (d) { return d.color; });
        legendItems.selectAll("text")
            .data(legendElements, function (d) { return d.label; })
            .call(function (d) { d.enter().append("text"); })
            .call(function (d) { d.exit().remove(); })
            .attr("y", function (d, i) { return i * Legend.separation + "px"; })
            .attr("x", (Legend.padding + Legend.boxSize) + "px")
            .text(function (d) { return d.label; });
        var legendBoundingBox = legendItems[0][0].getBBox();
        legendBox.attr("x", (legendBoundingBox.x - Legend.padding))
            .attr("y", (legendBoundingBox.y - Legend.padding))
            .attr("height", (legendBoundingBox.height + 2 * Legend.padding))
            .attr("width", (legendBoundingBox.width + 2 * Legend.padding))
            .style("fill", Legend.fillColor)
            .style("stroke", Legend.borderStrokeColor)
            .style("opacity", Legend.legendOpacity);
        svg.selectAll(".legend").attr("transform", "translate(" + Legend.offsetX + "," + Legend.offsetY + ")");
    });
    return Legend;
}());
var ChartStackedArea = (function (_super) {
    __extends(ChartStackedArea, _super);
    function ChartStackedArea(jsonStr) {
        var _this = _super.call(this, ComponentType.ChartStackedArea, jsonStr) || this;
        _this.render = function (appendToObject) {
            var nSeries = (!_this.xData ? 0 : _this.xData.length);
            var s = _this.getStyle();
            var margin = Style.getMargins(s);
            var xScale = d3.scale.linear().range([0, margin.widthExMargins]);
            var yScale = d3.scale.linear().range([margin.heightExMargins, 0]);
            var xAxis = d3.svg.axis().scale(xScale)
                .orient("bottom").ticks(5);
            if (_this.gridVerticalStrokeWidth != null && _this.gridVerticalStrokeWidth > 0) {
                xAxis.innerTickSize(-margin.heightExMargins);
            }
            var yAxis = d3.svg.axis().scale(yScale)
                .orient("left").ticks(5);
            if (_this.gridHorizontalStrokeWidth != null && _this.gridHorizontalStrokeWidth > 0) {
                yAxis.innerTickSize(-margin.widthExMargins);
            }
            if (_this.suppressAxisHorizontal === true)
                xAxis.tickValues([]);
            if (_this.suppressAxisVertical === true)
                yAxis.tickValues([]);
            var data = [];
            for (var i = 0; i < _this.xData.length; i++) {
                var obj = {};
                for (var j = 0; j < _this.labels.length; j++) {
                    obj[_this.labels[j]] = _this.yData[j][i];
                    obj['xValue'] = _this.xData[i];
                }
                data.push(obj);
            }
            var area = d3.svg.area()
                .x(function (d) { return xScale(d.xValue); })
                .y0(function (d) { return yScale(d.y0); })
                .y1(function (d) { return yScale(d.y0 + d.y); });
            var stack = d3.layout.stack()
                .values(function (d) { return d.values; });
            var svg = d3.select("#" + appendToObject.attr("id")).append("svg")
                .attr("width", margin.widthExMargins + margin.left + margin.right)
                .attr("height", margin.heightExMargins + margin.top + margin.bottom)
                .append("g")
                .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
            var color = d3.scale.category20();
            color.domain(d3.keys(data[0]).filter(function (key) {
                return key !== "xValue";
            }));
            var browsers = stack(color.domain().map(function (name) {
                return {
                    name: name,
                    values: data.map(function (d) {
                        return { xValue: d.xValue, y: d[name] * 1 };
                    })
                };
            }));
            var maxX = d3.max(data, function (d) {
                var vals = d3.keys(d).map(function (key) {
                    return key !== "xValue" ? d[key] : 0;
                });
                return d3.sum(vals);
            });
            xScale.domain(d3.extent(data, function (d) {
                return d.xValue;
            }));
            yScale.domain([0, maxX]);
            var browser = svg.selectAll(".browser")
                .data(browsers)
                .enter().append("g")
                .attr("class", "browser");
            var tempLabels = _this.labels;
            var defaultColor = d3.scale.category20();
            browser.append("path")
                .attr("class", "area")
                .attr("data-legend", function (d) { return d.name; })
                .attr("d", function (d) {
                return area(d.values);
            })
                .style("fill", function (d) {
                if (s && s.getSeriesColor(tempLabels.indexOf(d.name))) {
                    return s.getSeriesColor(tempLabels.indexOf(d.name));
                }
                else {
                    return defaultColor(String(tempLabels.indexOf(d.name)));
                }
            })
                .style({ "stroke-width": "0px" });
            var xAxisNode = svg.append("g")
                .attr("class", "x axis")
                .style("stroke", "#000")
                .style("stroke-width", (s != null && s.getAxisStrokeWidth() != null ? s.getAxisStrokeWidth() : ChartConstants.DEFAULT_AXIS_STROKE_WIDTH))
                .style("fill", "none")
                .attr("transform", "translate(0," + margin.heightExMargins + ")")
                .call(xAxis);
            xAxisNode.selectAll('text').style("stroke-width", 0).style("fill", "#000000");
            var yAxisNode = svg.append("g")
                .attr("class", "y axis")
                .style("stroke", "#000")
                .style("stroke-width", (s != null && s.getAxisStrokeWidth() != null ? s.getAxisStrokeWidth() : ChartConstants.DEFAULT_AXIS_STROKE_WIDTH))
                .style("fill", "none")
                .call(yAxis);
            yAxisNode.selectAll('text').style("stroke-width", 0).style("fill", "#000000");
            if (_this.title) {
                var titleStyle;
                if (_this.style)
                    titleStyle = _this.style.getTitleStyle();
                Chart.appendTitle(svg, _this.title, margin, titleStyle);
            }
            var legend = svg.append("g")
                .attr("class", "legend")
                .attr("transform", "translate(40,40)")
                .style("font-size", "12px")
                .call(Legend.legendFn);
        };
        var json = JSON.parse(jsonStr);
        if (!json["componentType"])
            json = json[ComponentType[ComponentType.ChartStackedArea]];
        _this.xData = json['x'];
        _this.yData = json['y'];
        _this.labels = json['labels'];
        return _this;
    }
    return ChartStackedArea;
}(Chart));
var ChartTimeline = (function (_super) {
    __extends(ChartTimeline, _super);
    function ChartTimeline(jsonStr) {
        var _this = _super.call(this, ComponentType.ChartTimeline, jsonStr) || this;
        _this.render = function (appendToObject) {
            var instance = _this;
            var s = _this.getStyle();
            var margin = Style.getMargins(s);
            _this.itemData = [];
            var count = 0;
            for (var i = 0; i < _this.laneData.length; i++) {
                for (var j = 0; j < _this.laneData[i].length; j++) {
                    var obj = {};
                    obj["start"] = _this.laneData[i][j]["startTimeMs"];
                    obj["end"] = _this.laneData[i][j]["endTimeMs"];
                    obj["id"] = count++;
                    obj["lane"] = i;
                    obj["color"] = _this.laneData[i][j]["color"];
                    obj["label"] = _this.laneData[i][j]["entryLabel"];
                    _this.itemData.push(obj);
                }
            }
            _this.lanes = [];
            for (var i = 0; i < _this.laneNames.length; i++) {
                var obj = {};
                obj["label"] = _this.laneNames[i];
                obj["id"] = i;
                _this.lanes.push(obj);
            }
            var svg = d3.select("#" + appendToObject.attr("id"))
                .append("svg")
                .style("stroke-width", (s && s.getStrokeWidth() ? s.getStrokeWidth() : ChartConstants.DEFAULT_CHART_STROKE_WIDTH))
                .style("fill", "none")
                .attr("width", s.getWidth())
                .attr("height", s.getHeight())
                .append("g");
            var heightExMargins = s.getHeight() - margin.top - margin.bottom;
            var widthExMargins = s.getWidth() - margin.left - margin.right;
            var miniHeight = _this.laneNames.length * ChartTimeline.MINI_LANE_HEIGHT_PX;
            var mainHeight = s.getHeight() - miniHeight - margin.top - margin.bottom - 25;
            var minTime = d3.min(_this.itemData, function (d) { return d.start; });
            var maxTime = d3.max(_this.itemData, function (d) { return d.end; });
            _this.x = d3.time.scale()
                .domain([minTime, maxTime])
                .range([0, widthExMargins]);
            _this.x1 = d3.time.scale().range([0, widthExMargins]);
            _this.y1 = d3.scale.linear().domain([0, _this.laneNames.length]).range([0, mainHeight]);
            _this.y2 = d3.scale.linear().domain([0, _this.laneNames.length]).range([0, miniHeight]);
            _this.rect = svg.append('defs').append('clipPath')
                .attr('id', 'clip')
                .append('rect')
                .attr('width', widthExMargins)
                .attr('height', s.getHeight() - 100);
            _this.mainView = svg.append('g')
                .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')')
                .attr('width', widthExMargins)
                .attr('height', mainHeight)
                .attr('font-size', '12px')
                .attr('font', 'sans-serif');
            _this.miniView = svg.append('g')
                .attr('transform', 'translate(' + margin.left + ',' + (mainHeight + margin.top + 25) + ')')
                .attr('width', widthExMargins)
                .attr('height', miniHeight)
                .attr('font-size', '10px')
                .attr('font', 'sans-serif');
            _this.mainView.append('g').selectAll('.laneLines')
                .data(_this.lanes)
                .enter().append('line')
                .attr('x1', 0)
                .attr('y1', function (d) {
                return d3.round(instance.y1(d.id)) + 0.5;
            })
                .attr('x2', widthExMargins)
                .attr('y2', function (d) {
                return d3.round(instance.y1(d.id)) + 0.5;
            })
                .attr('stroke', 'lightgray')
                .attr('stroke-width', 1);
            _this.mainView.append('g').selectAll('.laneText')
                .data(_this.lanes)
                .enter().append('text')
                .text(function (d) {
                if (d.label)
                    return d.label;
                return "";
            })
                .attr('x', -10)
                .attr('y', function (d) {
                return instance.y1(d.id + .5);
            })
                .attr('text-anchor', 'end')
                .attr("font", "8pt sans-serif")
                .attr('fill', 'black');
            _this.miniView.append('g').selectAll('.laneLines')
                .data(_this.lanes)
                .enter().append('line')
                .attr('x1', 0)
                .attr('y1', function (d) { return d3.round(instance.y2(d.id)) + 0.5; })
                .attr('x2', widthExMargins)
                .attr('y2', function (d) { return d3.round(instance.y2(d.id)) + 0.5; })
                .attr('stroke', 'gray')
                .attr('stroke-width', 1.0);
            _this.miniView.append('g').selectAll('.laneText')
                .data(_this.lanes)
                .enter().append('text')
                .text(function (d) {
                if (d.label)
                    return d.label;
                return "";
            })
                .attr('x', -10)
                .attr('y', function (d) {
                return instance.y2(d.id + .5);
            })
                .attr('dy', '0.5ex')
                .attr('text-anchor', 'end')
                .attr('fill', 'black');
            _this.xTimeAxis = d3.svg.axis()
                .scale(_this.x1)
                .orient('bottom')
                .ticks(d3.time.days, 1)
                .tickFormat(d3.time.format('%a %d'))
                .tickSize(6, 0);
            var temp = _this.mainView.append('g')
                .attr('transform', 'translate(0,' + mainHeight + ')')
                .attr('class', 'timeAxis')
                .attr('fill', 'black')
                .style("stroke", "black").style("stroke-width", 1.0).style("fill", "black")
                .attr("font", "10px sans-serif")
                .call(_this.xTimeAxis);
            temp.selectAll('text').style("stroke-width", 0.0).attr('stroke-width', 0.0);
            _this.itemRects = _this.mainView.append('g')
                .attr('clip-path', 'url(#clip)');
            _this.miniView.append('g').selectAll('miniItems')
                .data(_this.getMiniViewPaths(_this.itemData))
                .enter().append('path')
                .attr('class', function (d) {
                return 'miniItem ' + d["class"];
            })
                .attr('d', function (d) {
                return d.path;
            })
                .attr('stroke', 'black')
                .attr('stroke-width', 'black');
            _this.miniView.append('rect')
                .attr('pointer-events', 'painted')
                .attr('width', widthExMargins)
                .attr('height', miniHeight)
                .attr('visibility', 'hidden')
                .on('mouseup', _this.moveBrush);
            _this.brush = d3.svg.brush()
                .x(_this.x)
                .extent([minTime, maxTime])
                .on("brush", _this.renderChart);
            _this.miniView.append('g')
                .attr('class', 'x brush')
                .call(_this.brush)
                .selectAll('rect')
                .attr('y', 1)
                .attr('height', miniHeight - 1)
                .style('fill', 'gray')
                .style('fill-opacity', '0.2')
                .style('stroke', 'DarkSlateGray')
                .style('stroke-width', 1);
            _this.miniView.selectAll('rect.background').remove();
            _this.renderChart();
            if (_this.title) {
                var titleStyle;
                if (_this.style)
                    titleStyle = _this.style.getTitleStyle();
                var text = svg.append("text")
                    .text(_this.title)
                    .attr("x", (s.getWidth() / 2))
                    .attr("y", ((margin.top - 30) / 2))
                    .attr("text-anchor", "middle");
                if (titleStyle) {
                    if (titleStyle.getFont())
                        text.attr("font-family", titleStyle.getFont);
                    if (titleStyle.getFontSize() != null)
                        text.attr("font-size", titleStyle.getFontSize() + "pt");
                    if (titleStyle.getUnderline() != null)
                        text.style("text-decoration", "underline");
                    if (titleStyle.getColor())
                        text.style("fill", titleStyle.getColor);
                    else
                        text.style("fill", ChartConstants.DEFAULT_TITLE_COLOR);
                }
                else {
                    text.style("text-decoration", "underline");
                    text.style("fill", ChartConstants.DEFAULT_TITLE_COLOR);
                }
            }
        };
        _this.renderChart = function () {
            var instance = _this;
            var extent = _this.brush.extent();
            var minExtent = extent[0];
            var maxExtent = extent[1];
            var visibleItems = _this.itemData.filter(function (d) {
                return d.start < maxExtent && d.end > minExtent;
            });
            _this.miniView.select('.brush').call(_this.brush.extent([minExtent, maxExtent]));
            _this.x1.domain([minExtent, maxExtent]);
            var range = maxExtent - minExtent;
            if (range > 2 * ChartTimeline.MILLISEC_PER_WEEK) {
                _this.xTimeAxis.ticks(d3.time.mondays, 1).tickFormat(d3.time.format('%a %d'));
            }
            else if (range > 2 * ChartTimeline.MILLISEC_PER_DAY) {
                _this.xTimeAxis.ticks(d3.time.days, 1).tickFormat(d3.time.format('%a %d'));
            }
            else if (range > 2 * ChartTimeline.MILLISEC_PER_HOUR) {
                _this.xTimeAxis.ticks(d3.time.hours, 4).tickFormat(d3.time.format('%H %p'));
            }
            else if (range > 2 * ChartTimeline.MILLISEC_PER_MINUTE) {
                _this.xTimeAxis.ticks(d3.time.minutes, 1).tickFormat(d3.time.format('%H:%M'));
            }
            else if (range >= 30000) {
                _this.xTimeAxis.ticks(d3.time.seconds, 10).tickFormat(d3.time.format('%H:%M:%S'));
            }
            else {
                _this.xTimeAxis.ticks(d3.time.seconds, 1).tickFormat(d3.time.format('%H:%M:%S'));
            }
            _this.mainView.select('.timeAxis').call(_this.xTimeAxis);
            var rects = _this.itemRects.selectAll('rect')
                .data(visibleItems, function (d) { return d.id; })
                .attr('x', function (d) { return instance.x1(d.start); })
                .attr('width', function (d) { return instance.x1(d.end) - instance.x1(d.start); });
            rects.enter().append('rect')
                .attr('x', function (d) { return instance.x1(d.start); })
                .attr('y', function (d) { return instance.y1(d.lane) + ChartTimeline.ENTRY_LANE_HEIGHT_OFFSET_FRACTION * instance.y1(1) + 0.5; })
                .attr('width', function (d) { return instance.x1(d.end) - instance.x1(d.start); })
                .attr('height', function (d) { return ChartTimeline.ENTRY_LANE_HEIGHT_TOTAL_FRACTION * instance.y1(1); })
                .attr('stroke', 'black')
                .attr('fill', function (d) {
                if (d.color)
                    return d.color;
                return ChartTimeline.DEFAULT_COLOR;
            })
                .attr('stroke-width', 1);
            rects.exit().remove();
            var labels = _this.itemRects.selectAll('text')
                .data(visibleItems, function (d) {
                return d.id;
            })
                .attr('x', function (d) {
                return instance.x1(Math.max(d.start, minExtent)) + 2;
            })
                .attr('fill', 'black');
            labels.enter().append('text')
                .text(function (d) {
                if (instance.x1(d.end) - instance.x1(d.start) <= 30)
                    return "";
                if (d.label)
                    return d.label;
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
        _this.moveBrush = function () {
            var origin = d3.mouse(_this.rect[0]);
            var time = _this.x.invert(origin[0]).getTime();
            var halfExtent = (_this.brush.extent()[1].getTime() - _this.brush.extent()[0].getTime()) / 2;
            _this.brush.extent([new Date(time - halfExtent), new Date(time + halfExtent)]);
            _this.renderChart();
        };
        _this.getMiniViewPaths = function (items) {
            var paths = {}, d, offset = .5 * _this.y2(1) + 0.5, result = [];
            for (var i = 0; i < items.length; i++) {
                d = items[i];
                if (!paths[d["class"]])
                    paths[d["class"]] = '';
                paths[d["class"]] += ['M', _this.x(d.start), (_this.y2(d.lane) + offset), 'H', _this.x(d.end)].join(' ');
            }
            for (var className in paths) {
                result.push({ "class": className, path: paths[className] });
            }
            return result;
        };
        var json = JSON.parse(jsonStr);
        if (!json["componentType"])
            json = json[ComponentType[ComponentType.ChartTimeline]];
        _this.laneNames = json['laneNames'];
        _this.laneData = json['laneData'];
        return _this;
    }
    ChartTimeline.MINI_LANE_HEIGHT_PX = 12;
    ChartTimeline.ENTRY_LANE_HEIGHT_OFFSET_FRACTION = 0.05;
    ChartTimeline.ENTRY_LANE_HEIGHT_TOTAL_FRACTION = 0.90;
    ChartTimeline.MILLISEC_PER_MINUTE = 60 * 1000;
    ChartTimeline.MILLISEC_PER_HOUR = 60 * ChartTimeline.MILLISEC_PER_MINUTE;
    ChartTimeline.MILLISEC_PER_DAY = 24 * ChartTimeline.MILLISEC_PER_HOUR;
    ChartTimeline.MILLISEC_PER_WEEK = 7 * ChartTimeline.MILLISEC_PER_DAY;
    ChartTimeline.DEFAULT_COLOR = "LightGrey";
    return ChartTimeline;
}(Chart));
var StyleChart = (function (_super) {
    __extends(StyleChart, _super);
    function StyleChart(jsonObj) {
        var _this = _super.call(this, jsonObj['StyleChart']) || this;
        _this.getStrokeWidth = function () { return _this.strokeWidth; };
        _this.getPointSize = function () { return _this.pointSize; };
        _this.getSeriesColors = function () { return _this.seriesColors; };
        _this.getSeriesColor = function (idx) {
            if (!_this.seriesColors || idx < 0 || idx > _this.seriesColors.length)
                return null;
            return _this.seriesColors[idx];
        };
        _this.getAxisStrokeWidth = function () { return _this.axisStrokeWidth; };
        _this.getTitleStyle = function () { return _this.titleStyle; };
        var style = jsonObj['StyleChart'];
        if (style) {
            _this.strokeWidth = style['strokeWidth'];
            _this.pointSize = style['pointSize'];
            _this.seriesColors = style['seriesColors'];
            if (style['titleStyle'])
                _this.titleStyle = new StyleText(style['titleStyle']);
        }
        return _this;
    }
    return StyleChart;
}(Style));
var ComponentDiv = (function (_super) {
    __extends(ComponentDiv, _super);
    function ComponentDiv(jsonStr) {
        var _this = _super.call(this, ComponentType.ComponentDiv) || this;
        _this.render = function (appendToObject) {
            var newDiv = $('<div></div>');
            newDiv.uniqueId();
            if (_this.style) {
                if (_this.style.getWidth()) {
                    var unit = _this.style.getWidthUnit();
                    newDiv.width(_this.style.getWidth() + (unit ? unit : ""));
                }
                if (_this.style.getHeight()) {
                    var unit = _this.style.getHeightUnit();
                    newDiv.height(_this.style.getHeight() + (unit ? unit : ""));
                }
                if (_this.style.getBackgroundColor())
                    newDiv.css("background-color", _this.style.getBackgroundColor());
                if (_this.style.getFloatValue())
                    newDiv.css("float", _this.style.getFloatValue());
            }
            appendToObject.append(newDiv);
            if (_this.components) {
                for (var i = 0; i < _this.components.length; i++) {
                    _this.components[i].render(newDiv);
                }
            }
        };
        var json = JSON.parse(jsonStr);
        if (!json["componentType"])
            json = json[ComponentType[ComponentType.ComponentDiv]];
        var components = json['components'];
        if (components) {
            _this.components = [];
            for (var i = 0; i < components.length; i++) {
                var asStr = JSON.stringify(components[i]);
                _this.components.push(Component.getComponent(asStr));
            }
        }
        if (json['style'])
            _this.style = new StyleDiv(json['style']);
        return _this;
    }
    return ComponentDiv;
}(Component));
var StyleDiv = (function (_super) {
    __extends(StyleDiv, _super);
    function StyleDiv(jsonObj) {
        var _this = _super.call(this, jsonObj['StyleDiv']) || this;
        _this.getFloatValue = function () { return _this.floatValue; };
        if (jsonObj && jsonObj['StyleDiv'])
            _this.floatValue = jsonObj['StyleDiv']['floatValue'];
        return _this;
    }
    return StyleDiv;
}(Style));
var DecoratorAccordion = (function (_super) {
    __extends(DecoratorAccordion, _super);
    function DecoratorAccordion(jsonStr) {
        var _this = _super.call(this, ComponentType.DecoratorAccordion) || this;
        _this.render = function (appendToObject) {
            var s = _this.style;
            var outerDiv = $('<div></div>');
            outerDiv.uniqueId();
            var titleDiv;
            if (_this.title)
                titleDiv = $('<div>' + _this.title + '</div>');
            else
                titleDiv = $('<div></div>');
            titleDiv.uniqueId();
            outerDiv.append(titleDiv);
            var innerDiv = $('<div></div>');
            innerDiv.uniqueId();
            outerDiv.append(innerDiv);
            if (_this.innerComponents) {
                for (var i = 0; i < _this.innerComponents.length; i++) {
                    _this.innerComponents[i].render(innerDiv);
                }
            }
            appendToObject.append(outerDiv);
            if (_this.defaultCollapsed)
                outerDiv.accordion({ collapsible: true, heightStyle: "content", active: false });
            else
                outerDiv.accordion({ collapsible: true, heightStyle: "content" });
        };
        var json = JSON.parse(jsonStr);
        if (!json["componentType"])
            json = json[ComponentType[ComponentType.DecoratorAccordion]];
        _this.title = json['title'];
        _this.defaultCollapsed = json['defaultCollapsed'];
        var innerCs = json['innerComponents'];
        if (innerCs) {
            _this.innerComponents = [];
            for (var i = 0; i < innerCs.length; i++) {
                var asStr = JSON.stringify(innerCs[i]);
                _this.innerComponents.push(Component.getComponent(asStr));
            }
        }
        if (json['style'])
            _this.style = new StyleAccordion(json['style']);
        return _this;
    }
    return DecoratorAccordion;
}(Component));
var StyleAccordion = (function (_super) {
    __extends(StyleAccordion, _super);
    function StyleAccordion(jsonObj) {
        return _super.call(this, jsonObj['StyleAccordion']) || this;
    }
    return StyleAccordion;
}(Style));
var ComponentTable = (function (_super) {
    __extends(ComponentTable, _super);
    function ComponentTable(jsonStr) {
        var _this = _super.call(this, ComponentType.ComponentTable) || this;
        _this.render = function (appendToObject) {
            var s = _this.style;
            var margin = Style.getMargins(s);
            var tbl = document.createElement('table');
            tbl.style.width = '100%';
            if (s && s.getBorderWidthPx() != null)
                tbl.setAttribute('border', String(s.getBorderWidthPx()));
            if (s && s.getBackgroundColor())
                tbl.style.backgroundColor = s.getBackgroundColor();
            if (s && s.getWhitespaceMode())
                tbl.style.whiteSpace = s.getWhitespaceMode();
            if (s && s.getColumnWidths()) {
                var colWidths = s.getColumnWidths();
                var unit = TSUtils.normalizeLengthUnit(s.getColumnWidthUnit());
                for (var i = 0; i < colWidths.length; i++) {
                    var col = document.createElement('col');
                    col.setAttribute('width', colWidths[i] + unit);
                    tbl.appendChild(col);
                }
            }
            var padTop = 1;
            var padRight = 1;
            var padBottom = 1;
            var padLeft = 1;
            if (_this.header) {
                var theader = document.createElement('thead');
                var headerRow = document.createElement('tr');
                if (s && s.getHeaderColor())
                    headerRow.style.backgroundColor = s.getHeaderColor();
                for (var i = 0; i < _this.header.length; i++) {
                    var headerd = document.createElement('th');
                    headerd.style.padding = padTop + 'px ' + padRight + 'px ' + padBottom + 'px ' + padLeft + 'px';
                    headerd.appendChild(document.createTextNode(_this.header[i]));
                    headerRow.appendChild(headerd);
                }
                tbl.appendChild(headerRow);
            }
            if (_this.content) {
                var tbdy = document.createElement('tbody');
                for (var i = 0; i < _this.content.length; i++) {
                    var tr = document.createElement('tr');
                    for (var j = 0; j < _this.content[i].length; j++) {
                        var td = document.createElement('td');
                        td.style.padding = padTop + 'px ' + padRight + 'px ' + padBottom + 'px ' + padLeft + 'px';
                        td.appendChild(document.createTextNode(_this.content[i][j]));
                        tr.appendChild(td);
                    }
                    tbdy.appendChild(tr);
                }
                tbl.appendChild(tbdy);
            }
            appendToObject.append(tbl);
        };
        var json = JSON.parse(jsonStr);
        if (!json["componentType"])
            json = json[ComponentType[ComponentType.ComponentTable]];
        _this.header = json['header'];
        _this.content = json['content'];
        if (json['style'])
            _this.style = new StyleTable(json['style']);
        return _this;
    }
    return ComponentTable;
}(Component));
var StyleTable = (function (_super) {
    __extends(StyleTable, _super);
    function StyleTable(jsonObj) {
        var _this = _super.call(this, jsonObj['StyleTable']) || this;
        _this.getColumnWidths = function () { return _this.columnWidths; };
        _this.getColumnWidthUnit = function () { return _this.columnWidthUnit; };
        _this.getBorderWidthPx = function () { return _this.borderWidthPx; };
        _this.getHeaderColor = function () { return _this.headerColor; };
        _this.getWhitespaceMode = function () { return _this.whitespaceMode; };
        var style = jsonObj['StyleTable'];
        if (style) {
            _this.columnWidths = jsonObj['StyleTable']['columnWidths'];
            _this.borderWidthPx = jsonObj['StyleTable']['borderWidthPx'];
            _this.headerColor = jsonObj['StyleTable']['headerColor'];
            _this.columnWidthUnit = jsonObj['StyleTable']['columnWidthUnit'];
            _this.whitespaceMode = jsonObj['StyleTable']['whitespaceMode'];
        }
        return _this;
    }
    return StyleTable;
}(Style));
var ComponentText = (function (_super) {
    __extends(ComponentText, _super);
    function ComponentText(jsonStr) {
        var _this = _super.call(this, ComponentType.ComponentText) || this;
        _this.render = function (appendToObject) {
            var textNode = document.createTextNode(_this.text);
            if (_this.style) {
                var newSpan = document.createElement('span');
                if (_this.style.getFont())
                    newSpan.style.font = _this.style.getFont();
                if (_this.style.getFontSize() != null)
                    newSpan.style.fontSize = _this.style.getFontSize() + "pt";
                if (_this.style.getUnderline() != null)
                    newSpan.style.textDecoration = 'underline';
                if (_this.style.getColor())
                    newSpan.style.color = _this.style.getColor();
                if (_this.style.getMarginTop())
                    newSpan.style.marginTop = _this.style.getMarginTop() + "px";
                if (_this.style.getMarginBottom())
                    newSpan.style.marginBottom = _this.style.getMarginBottom() + "px";
                if (_this.style.getMarginLeft())
                    newSpan.style.marginLeft = _this.style.getMarginLeft() + "px";
                if (_this.style.getMarginRight())
                    newSpan.style.marginRight = _this.style.getMarginRight() + "px";
                if (_this.style.getWhitespacePre())
                    newSpan.style.whiteSpace = 'pre';
                newSpan.appendChild(textNode);
                appendToObject.append(newSpan);
            }
            else {
                var newSpan = document.createElement('span');
                newSpan.appendChild(textNode);
                appendToObject.append(newSpan);
            }
        };
        var json = JSON.parse(jsonStr);
        if (!json["componentType"])
            json = json[ComponentType[ComponentType.ComponentText]];
        _this.text = json['text'];
        if (json['style'])
            _this.style = new StyleText(json['style']);
        return _this;
    }
    return ComponentText;
}(Component));
var StyleText = (function (_super) {
    __extends(StyleText, _super);
    function StyleText(jsonObj) {
        var _this = _super.call(this, jsonObj['StyleText']) || this;
        _this.getFont = function () { return _this.font; };
        _this.getFontSize = function () { return _this.fontSize; };
        _this.getUnderline = function () { return _this.underline; };
        _this.getColor = function () { return _this.color; };
        _this.getWhitespacePre = function () { return _this.whitespacePre; };
        var style = jsonObj['StyleText'];
        if (style) {
            _this.font = style['font'];
            _this.fontSize = style['fontSize'];
            _this.underline = style['underline'];
            _this.color = style['color'];
            _this.whitespacePre = style['whitespacePre'];
        }
        return _this;
    }
    return StyleText;
}(Style));
//# sourceMappingURL=dl4j-ui.js.map