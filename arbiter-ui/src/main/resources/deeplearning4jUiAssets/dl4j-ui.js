var __extends = (this && this.__extends) || function (d, b) {
    for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p];
    function __() { this.constructor = d; }
    d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
};
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
        _super.call(this, componentType);
        var jsonOrig = JSON.parse(jsonStr);
        var json = JSON.parse(jsonStr);
        if (!json["componentType"])
            json = json[ComponentType[componentType]];
        this.suppressAxisHorizontal = json['suppressAxisHorizontal'];
        this.suppressAxisVertical = json['suppressAxisVertical'];
        this.showLegend = json['showLegend'];
        this.title = json['title'];
        this.setXMin = json['setXMin'];
        this.setXMax = json['setXMax'];
        this.setYMin = json['setYMin'];
        this.setYMax = json['setYMax'];
        this.gridVerticalStrokeWidth = json['gridVerticalStrokeWidth'];
        this.gridHorizontalStrokeWidth = json['gridHorizontalStrokeWidth'];
        if (json['style'])
            this.style = new StyleChart(json['style']);
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
        _super.call(this, ComponentType.ChartHistogram, jsonStr);
        this.render = function (appendToObject) {
            var s = this.getStyle();
            var margin = Style.getMargins(s);
            var xMin;
            var xMax;
            var yMin;
            var yMax;
            if (this.setXMin)
                xMin = this.setXMin;
            else
                xMin = (this.lowerBounds ? d3.min(this.lowerBounds) : 0);
            if (this.setXMax)
                xMax = this.setXMax;
            else
                xMax = (this.upperBounds ? d3.max(this.upperBounds) : 1);
            if (this.setYMin)
                yMin = this.setYMin;
            else
                yMin = 0;
            if (this.setYMax)
                yMax = this.setYMax;
            else
                yMax = (this.yValues ? d3.max(this.yValues) : 1);
            var xScale = d3.scale.linear()
                .domain([xMin, xMax])
                .range([0, margin.widthExMargins]);
            var xAxis = d3.svg.axis().scale(xScale)
                .orient("bottom").ticks(5);
            if (this.gridVerticalStrokeWidth && this.gridVerticalStrokeWidth > 0) {
                xAxis.innerTickSize(-margin.heightExMargins);
            }
            var yScale = d3.scale.linear()
                .domain([0, yMax])
                .range([margin.heightExMargins, 0]);
            var yAxis = d3.svg.axis().scale(yScale)
                .orient("left").ticks(5);
            if (this.gridHorizontalStrokeWidth && this.gridHorizontalStrokeWidth > 0) {
                yAxis.innerTickSize(-margin.widthExMargins);
            }
            if (this.suppressAxisHorizontal === true)
                xAxis.tickValues([]);
            if (this.suppressAxisVertical === true)
                yAxis.tickValues([]);
            var lowerBounds = this.lowerBounds;
            var upperBounds = this.upperBounds;
            var yValues = this.yValues;
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
            if (this.gridVerticalStrokeWidth != null)
                xAxisNode.selectAll('.axis line').style({ 'stroke-width': this.gridVerticalStrokeWidth });
            var yAxisNode = svg.append("g")
                .attr("class", "y axis")
                .style("stroke", "#000")
                .style("stroke-width", (s != null && s.getAxisStrokeWidth() != null ? s.getAxisStrokeWidth() : ChartConstants.DEFAULT_AXIS_STROKE_WIDTH))
                .style("fill", "none")
                .call(yAxis);
            yAxisNode.selectAll('text').style("stroke-width", 0).style("fill", "#000000");
            if (this.gridHorizontalStrokeWidth != null)
                yAxisNode.selectAll('.axis line').style({ 'stroke-width': this.gridHorizontalStrokeWidth });
            if (this.title) {
                var titleStyle;
                if (this.style)
                    titleStyle = this.style.getTitleStyle();
                Chart.appendTitle(svg, this.title, margin, titleStyle);
            }
        };
        var json = JSON.parse(jsonStr);
        if (!json["componentType"])
            json = json[ComponentType[ComponentType.ChartHistogram]];
        this.lowerBounds = json['lowerBounds'];
        this.upperBounds = json['upperBounds'];
        this.yValues = json['yvalues'];
    }
    return ChartHistogram;
}(Chart));
var ChartLine = (function (_super) {
    __extends(ChartLine, _super);
    function ChartLine(jsonStr) {
        _super.call(this, ComponentType.ChartLine, jsonStr);
        this.render = function (appendToObject) {
            var nSeries = (!this.xData ? 0 : this.xData.length);
            var s = this.getStyle();
            var margin = Style.getMargins(s);
            var xScale = d3.scale.linear().range([0, margin.widthExMargins]);
            var yScale = d3.scale.linear().range([margin.heightExMargins, 0]);
            var xAxis = d3.svg.axis().scale(xScale)
                .orient("bottom").ticks(5);
            if (this.gridVerticalStrokeWidth != null && this.gridVerticalStrokeWidth > 0) {
                xAxis.innerTickSize(-margin.heightExMargins);
            }
            var yAxis = d3.svg.axis().scale(yScale)
                .orient("left").ticks(5);
            if (this.gridHorizontalStrokeWidth != null && this.gridHorizontalStrokeWidth > 0) {
                yAxis.innerTickSize(-margin.widthExMargins);
            }
            if (this.suppressAxisHorizontal === true)
                xAxis.tickValues([]);
            if (this.suppressAxisVertical === true)
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
            if (this.setXMin != null)
                xMin = this.setXMin;
            else
                xMin = (this.xData ? TSUtils.min(this.xData) : 0);
            if (this.setXMax != null)
                xMax = this.setXMax;
            else
                xMax = (this.xData ? TSUtils.max(this.xData) : 1);
            if (this.setYMin != null)
                yMin = this.setYMin;
            else
                yMin = (this.yData ? TSUtils.min(this.yData) : 0);
            if (this.setYMax != null)
                yMax = this.setYMax;
            else
                yMax = (this.yData ? TSUtils.max(this.yData) : 1);
            xScale.domain([xMin, xMax]);
            yScale.domain([yMin, yMax]);
            var defaultColor = d3.scale.category10();
            for (var i = 0; i < nSeries; i++) {
                var xVals = this.xData[i];
                var yVals = this.yData[i];
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
            if (this.gridVerticalStrokeWidth != null)
                xAxisNode.selectAll('.axis line').style({ 'stroke-width': this.gridVerticalStrokeWidth });
            var yAxisNode = svg.append("g")
                .attr("class", "y axis")
                .style("stroke", "#000")
                .style("stroke-width", (s != null && s.getAxisStrokeWidth() != null ? s.getAxisStrokeWidth() : ChartConstants.DEFAULT_AXIS_STROKE_WIDTH))
                .style("fill", "none")
                .call(yAxis);
            yAxisNode.selectAll('text').style("stroke-width", 0).style("fill", "#000000");
            if (this.gridHorizontalStrokeWidth != null)
                yAxisNode.selectAll('.axis line').style({ 'stroke-width': this.gridHorizontalStrokeWidth });
            if (this.seriesNames && this.showLegend === true) {
                var legendSpace = margin.widthExMargins / i;
                for (var i = 0; i < nSeries; i++) {
                    var values = this.xData[i];
                    var yValues = this.yData[i];
                    var lastX = values[values.length - 1];
                    var lastY = yValues[yValues.length - 1];
                    var toDisplay = this.seriesNames[i];
                    svg.append("text")
                        .attr("x", (legendSpace / 2) + i * legendSpace)
                        .attr("y", margin.heightExMargins + (margin.bottom / 2) + 5)
                        .attr("class", "legend")
                        .style("fill", (s && s.getSeriesColor(i) ? s.getSeriesColor(i) : defaultColor(String(i))))
                        .text(toDisplay);
                }
            }
            if (this.title) {
                var titleStyle;
                if (this.style)
                    titleStyle = this.style.getTitleStyle();
                Chart.appendTitle(svg, this.title, margin, titleStyle);
            }
        };
        var json = JSON.parse(jsonStr);
        if (!json["componentType"])
            json = json[ComponentType[ComponentType.ChartLine]];
        this.xData = json['x'];
        this.yData = json['y'];
        this.seriesNames = json['seriesNames'];
    }
    return ChartLine;
}(Chart));
var ChartScatter = (function (_super) {
    __extends(ChartScatter, _super);
    function ChartScatter(jsonStr) {
        _super.call(this, ComponentType.ChartScatter, jsonStr);
        this.render = function (appendToObject) {
            var nSeries = (!this.xData ? 0 : this.xData.length);
            var s = this.getStyle();
            var margin = Style.getMargins(s);
            var xScale = d3.scale.linear().range([0, margin.widthExMargins]);
            var yScale = d3.scale.linear().range([margin.heightExMargins, 0]);
            var xAxis = d3.svg.axis().scale(xScale)
                .innerTickSize(-margin.heightExMargins)
                .orient("bottom").ticks(5);
            var yAxis = d3.svg.axis().scale(yScale)
                .innerTickSize(-margin.widthExMargins)
                .orient("left").ticks(5);
            if (this.suppressAxisHorizontal === true)
                xAxis.tickValues([]);
            if (this.suppressAxisVertical === true)
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
            if (this.setXMin)
                xMin = this.setXMin;
            else
                xMin = (this.xData ? TSUtils.min(this.xData) : 0);
            if (this.setXMax)
                xMax = this.setXMax;
            else
                xMax = (this.xData ? TSUtils.max(this.xData) : 1);
            if (this.setYMin)
                yMin = this.setYMin;
            else
                yMin = (this.yData ? TSUtils.min(this.yData) : 0);
            if (this.setYMax)
                yMax = this.setYMax;
            else
                yMax = (this.yData ? TSUtils.max(this.yData) : 1);
            xScale.domain([xMin, xMax]);
            yScale.domain([yMin, yMax]);
            var defaultColor = d3.scale.category10();
            for (var i = 0; i < nSeries; i++) {
                var xVals = this.xData[i];
                var yVals = this.yData[i];
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
            if (this.gridVerticalStrokeWidth != null)
                xAxisNode.selectAll('.axis line').style({ 'stroke-width': this.gridVerticalStrokeWidth });
            var yAxisNode = svg.append("g")
                .attr("class", "y axis")
                .style("stroke", "#000")
                .style("stroke-width", (s != null && s.getAxisStrokeWidth() != null ? s.getAxisStrokeWidth() : ChartConstants.DEFAULT_AXIS_STROKE_WIDTH))
                .style("fill", "none")
                .call(yAxis);
            yAxisNode.selectAll('text').style("stroke-width", 0).style("fill", "#000000");
            if (this.gridHorizontalStrokeWidth != null)
                yAxisNode.selectAll('.axis line').style({ 'stroke-width': this.gridHorizontalStrokeWidth });
            if (this.seriesNames && this.showLegend === true) {
                var legendSpace = margin.widthExMargins / i;
                for (var i = 0; i < nSeries; i++) {
                    var values = this.xData[i];
                    var yValues = this.yData[i];
                    var lastX = values[values.length - 1];
                    var lastY = yValues[yValues.length - 1];
                    var toDisplay;
                    if (!lastX || !lastY)
                        toDisplay = this.seriesNames[i] + " (no data)";
                    else
                        toDisplay = this.seriesNames[i] + " (" + lastX.toPrecision(5) + "," + lastY.toPrecision(5) + ")";
                    svg.append("text")
                        .attr("x", (legendSpace / 2) + i * legendSpace)
                        .attr("y", margin.heightExMargins + (margin.bottom / 2) + 5)
                        .attr("class", "legend")
                        .style("fill", (s && s.getSeriesColor(i) ? s.getSeriesColor(i) : defaultColor(String(i))))
                        .text(toDisplay);
                }
            }
            if (this.title) {
                var titleStyle;
                if (this.style)
                    titleStyle = this.style.getTitleStyle();
                Chart.appendTitle(svg, this.title, margin, titleStyle);
            }
        };
        var json = JSON.parse(jsonStr);
        if (!json["componentType"])
            json = json[ComponentType[ComponentType.ChartScatter]];
        this.xData = json['x'];
        this.yData = json['y'];
        this.seriesNames = json['seriesNames'];
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
        _super.call(this, ComponentType.ChartStackedArea, jsonStr);
        this.render = function (appendToObject) {
            var nSeries = (!this.xData ? 0 : this.xData.length);
            var s = this.getStyle();
            var margin = Style.getMargins(s);
            var xScale = d3.scale.linear().range([0, margin.widthExMargins]);
            var yScale = d3.scale.linear().range([margin.heightExMargins, 0]);
            var xAxis = d3.svg.axis().scale(xScale)
                .orient("bottom").ticks(5);
            if (this.gridVerticalStrokeWidth != null && this.gridVerticalStrokeWidth > 0) {
                xAxis.innerTickSize(-margin.heightExMargins);
            }
            var yAxis = d3.svg.axis().scale(yScale)
                .orient("left").ticks(5);
            if (this.gridHorizontalStrokeWidth != null && this.gridHorizontalStrokeWidth > 0) {
                yAxis.innerTickSize(-margin.widthExMargins);
            }
            if (this.suppressAxisHorizontal === true)
                xAxis.tickValues([]);
            if (this.suppressAxisVertical === true)
                yAxis.tickValues([]);
            var data = [];
            for (var i = 0; i < this.xData.length; i++) {
                var obj = {};
                for (var j = 0; j < this.labels.length; j++) {
                    obj[this.labels[j]] = this.yData[j][i];
                    obj['xValue'] = this.xData[i];
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
            var tempLabels = this.labels;
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
            if (this.title) {
                var titleStyle;
                if (this.style)
                    titleStyle = this.style.getTitleStyle();
                Chart.appendTitle(svg, this.title, margin, titleStyle);
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
        this.xData = json['x'];
        this.yData = json['y'];
        this.labels = json['labels'];
    }
    return ChartStackedArea;
}(Chart));
var ChartTimeline = (function (_super) {
    __extends(ChartTimeline, _super);
    function ChartTimeline(jsonStr) {
        _super.call(this, ComponentType.ChartTimeline, jsonStr);
        this.render = function (appendToObject) {
            var instance = this;
            var s = this.getStyle();
            var margin = Style.getMargins(s);
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
            var svg = d3.select("#" + appendToObject.attr("id"))
                .append("svg")
                .style("stroke-width", (s && s.getStrokeWidth() ? s.getStrokeWidth() : ChartConstants.DEFAULT_CHART_STROKE_WIDTH))
                .style("fill", "none")
                .attr("width", s.getWidth())
                .attr("height", s.getHeight())
                .append("g");
            var heightExMargins = s.getHeight() - margin.top - margin.bottom;
            var widthExMargins = s.getWidth() - margin.left - margin.right;
            var miniHeight = this.laneNames.length * ChartTimeline.MINI_LANE_HEIGHT_PX;
            var mainHeight = s.getHeight() - miniHeight - margin.top - margin.bottom - 25;
            var minTime = d3.min(this.itemData, function (d) { return d.start; });
            var maxTime = d3.max(this.itemData, function (d) { return d.end; });
            this.x = d3.time.scale()
                .domain([minTime, maxTime])
                .range([0, widthExMargins]);
            this.x1 = d3.time.scale().range([0, widthExMargins]);
            this.y1 = d3.scale.linear().domain([0, this.laneNames.length]).range([0, mainHeight]);
            this.y2 = d3.scale.linear().domain([0, this.laneNames.length]).range([0, miniHeight]);
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
                .attr('transform', 'translate(' + margin.left + ',' + (mainHeight + margin.top + 25) + ')')
                .attr('width', widthExMargins)
                .attr('height', miniHeight)
                .attr('font-size', '10px')
                .attr('font', 'sans-serif');
            this.mainView.append('g').selectAll('.laneLines')
                .data(this.lanes)
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
            this.mainView.append('g').selectAll('.laneText')
                .data(this.lanes)
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
            this.miniView.append('g').selectAll('.laneLines')
                .data(this.lanes)
                .enter().append('line')
                .attr('x1', 0)
                .attr('y1', function (d) { return d3.round(instance.y2(d.id)) + 0.5; })
                .attr('x2', widthExMargins)
                .attr('y2', function (d) { return d3.round(instance.y2(d.id)) + 0.5; })
                .attr('stroke', 'gray')
                .attr('stroke-width', 1.0);
            this.miniView.append('g').selectAll('.laneText')
                .data(this.lanes)
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
            this.xTimeAxis = d3.svg.axis()
                .scale(this.x1)
                .orient('bottom')
                .ticks(d3.time.days, 1)
                .tickFormat(d3.time.format('%a %d'))
                .tickSize(6, 0);
            var temp = this.mainView.append('g')
                .attr('transform', 'translate(0,' + mainHeight + ')')
                .attr('class', 'timeAxis')
                .attr('fill', 'black')
                .style("stroke", "black").style("stroke-width", 1.0).style("fill", "black")
                .attr("font", "10px sans-serif")
                .call(this.xTimeAxis);
            temp.selectAll('text').style("stroke-width", 0.0).attr('stroke-width', 0.0);
            this.itemRects = this.mainView.append('g')
                .attr('clip-path', 'url(#clip)');
            this.miniView.append('g').selectAll('miniItems')
                .data(this.getMiniViewPaths(this.itemData))
                .enter().append('path')
                .attr('class', function (d) {
                return 'miniItem ' + d.class;
            })
                .attr('d', function (d) {
                return d.path;
            })
                .attr('stroke', 'black')
                .attr('stroke-width', 'black');
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
                .style('fill', 'gray')
                .style('fill-opacity', '0.2')
                .style('stroke', 'DarkSlateGray')
                .style('stroke-width', 1);
            this.miniView.selectAll('rect.background').remove();
            this.renderChart();
            if (this.title) {
                var titleStyle;
                if (this.style)
                    titleStyle = this.style.getTitleStyle();
                var text = svg.append("text")
                    .text(this.title)
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
        this.renderChart = function () {
            var instance = this;
            var extent = this.brush.extent();
            var minExtent = extent[0];
            var maxExtent = extent[1];
            var visibleItems = this.itemData.filter(function (d) {
                return d.start < maxExtent && d.end > minExtent;
            });
            this.miniView.select('.brush').call(this.brush.extent([minExtent, maxExtent]));
            this.x1.domain([minExtent, maxExtent]);
            var range = maxExtent - minExtent;
            if (range > 2 * ChartTimeline.MILLISEC_PER_WEEK) {
                this.xTimeAxis.ticks(d3.time.mondays, 1).tickFormat(d3.time.format('%a %d'));
            }
            else if (range > 2 * ChartTimeline.MILLISEC_PER_DAY) {
                this.xTimeAxis.ticks(d3.time.days, 1).tickFormat(d3.time.format('%a %d'));
            }
            else if (range > 2 * ChartTimeline.MILLISEC_PER_HOUR) {
                this.xTimeAxis.ticks(d3.time.hours, 4).tickFormat(d3.time.format('%H %p'));
            }
            else if (range > 2 * ChartTimeline.MILLISEC_PER_MINUTE) {
                this.xTimeAxis.ticks(d3.time.minutes, 1).tickFormat(d3.time.format('%H:%M'));
            }
            else if (range >= 30000) {
                this.xTimeAxis.ticks(d3.time.seconds, 10).tickFormat(d3.time.format('%H:%M:%S'));
            }
            else {
                this.xTimeAxis.ticks(d3.time.seconds, 1).tickFormat(d3.time.format('%H:%M:%S'));
            }
            this.mainView.select('.timeAxis').call(this.xTimeAxis);
            var rects = this.itemRects.selectAll('rect')
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
            var labels = this.itemRects.selectAll('text')
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
        this.moveBrush = function () {
            var origin = d3.mouse(this.rect[0]);
            var time = this.x.invert(origin[0]).getTime();
            var halfExtent = (this.brush.extent()[1].getTime() - this.brush.extent()[0].getTime()) / 2;
            this.brush.extent([new Date(time - halfExtent), new Date(time + halfExtent)]);
            this.renderChart();
        };
        this.getMiniViewPaths = function (items) {
            var paths = {}, d, offset = .5 * this.y2(1) + 0.5, result = [];
            for (var i = 0; i < items.length; i++) {
                d = items[i];
                if (!paths[d.class])
                    paths[d.class] = '';
                paths[d.class] += ['M', this.x(d.start), (this.y2(d.lane) + offset), 'H', this.x(d.end)].join(' ');
            }
            for (var className in paths) {
                result.push({ class: className, path: paths[className] });
            }
            return result;
        };
        var json = JSON.parse(jsonStr);
        if (!json["componentType"])
            json = json[ComponentType[ComponentType.ChartTimeline]];
        this.laneNames = json['laneNames'];
        this.laneData = json['laneData'];
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
        var _this = this;
        _super.call(this, jsonObj['StyleChart']);
        this.getStrokeWidth = function () { return _this.strokeWidth; };
        this.getPointSize = function () { return _this.pointSize; };
        this.getSeriesColors = function () { return _this.seriesColors; };
        this.getSeriesColor = function (idx) {
            if (!this.seriesColors || idx < 0 || idx > this.seriesColors.length)
                return null;
            return _this.seriesColors[idx];
        };
        this.getAxisStrokeWidth = function () { return _this.axisStrokeWidth; };
        this.getTitleStyle = function () { return _this.titleStyle; };
        var style = jsonObj['StyleChart'];
        if (style) {
            this.strokeWidth = style['strokeWidth'];
            this.pointSize = style['pointSize'];
            this.seriesColors = style['seriesColors'];
            if (style['titleStyle'])
                this.titleStyle = new StyleText(style['titleStyle']);
        }
    }
    return StyleChart;
}(Style));
var ComponentDiv = (function (_super) {
    __extends(ComponentDiv, _super);
    function ComponentDiv(jsonStr) {
        _super.call(this, ComponentType.ComponentDiv);
        this.render = function (appendToObject) {
            var newDiv = $('<div></div>');
            newDiv.uniqueId();
            if (this.style) {
                if (this.style.getWidth()) {
                    var unit = this.style.getWidthUnit();
                    newDiv.width(this.style.getWidth() + (unit ? unit : ""));
                }
                if (this.style.getHeight()) {
                    var unit = this.style.getHeightUnit();
                    newDiv.height(this.style.getHeight() + (unit ? unit : ""));
                }
                if (this.style.getBackgroundColor())
                    newDiv.css("background-color", this.style.getBackgroundColor());
                if (this.style.getFloatValue())
                    newDiv.css("float", this.style.getFloatValue());
            }
            appendToObject.append(newDiv);
            if (this.components) {
                for (var i = 0; i < this.components.length; i++) {
                    this.components[i].render(newDiv);
                }
            }
        };
        var json = JSON.parse(jsonStr);
        if (!json["componentType"])
            json = json[ComponentType[ComponentType.ComponentDiv]];
        var components = json['components'];
        if (components) {
            this.components = [];
            for (var i = 0; i < components.length; i++) {
                var asStr = JSON.stringify(components[i]);
                this.components.push(Component.getComponent(asStr));
            }
        }
        if (json['style'])
            this.style = new StyleDiv(json['style']);
    }
    return ComponentDiv;
}(Component));
var StyleDiv = (function (_super) {
    __extends(StyleDiv, _super);
    function StyleDiv(jsonObj) {
        var _this = this;
        _super.call(this, jsonObj['StyleDiv']);
        this.getFloatValue = function () { return _this.floatValue; };
        if (jsonObj && jsonObj['StyleDiv'])
            this.floatValue = jsonObj['StyleDiv']['floatValue'];
    }
    return StyleDiv;
}(Style));
var DecoratorAccordion = (function (_super) {
    __extends(DecoratorAccordion, _super);
    function DecoratorAccordion(jsonStr) {
        _super.call(this, ComponentType.DecoratorAccordion);
        this.render = function (appendToObject) {
            var s = this.style;
            var outerDiv = $('<div></div>');
            outerDiv.uniqueId();
            var titleDiv;
            if (this.title)
                titleDiv = $('<div>' + this.title + '</div>');
            else
                titleDiv = $('<div></div>');
            titleDiv.uniqueId();
            outerDiv.append(titleDiv);
            var innerDiv = $('<div></div>');
            innerDiv.uniqueId();
            outerDiv.append(innerDiv);
            if (this.innerComponents) {
                for (var i = 0; i < this.innerComponents.length; i++) {
                    this.innerComponents[i].render(innerDiv);
                }
            }
            appendToObject.append(outerDiv);
            if (this.defaultCollapsed)
                outerDiv.accordion({ collapsible: true, heightStyle: "content", active: false });
            else
                outerDiv.accordion({ collapsible: true, heightStyle: "content" });
        };
        var json = JSON.parse(jsonStr);
        if (!json["componentType"])
            json = json[ComponentType[ComponentType.DecoratorAccordion]];
        this.title = json['title'];
        this.defaultCollapsed = json['defaultCollapsed'];
        var innerCs = json['innerComponents'];
        if (innerCs) {
            this.innerComponents = [];
            for (var i = 0; i < innerCs.length; i++) {
                var asStr = JSON.stringify(innerCs[i]);
                this.innerComponents.push(Component.getComponent(asStr));
            }
        }
        if (json['style'])
            this.style = new StyleAccordion(json['style']);
    }
    return DecoratorAccordion;
}(Component));
var StyleAccordion = (function (_super) {
    __extends(StyleAccordion, _super);
    function StyleAccordion(jsonObj) {
        _super.call(this, jsonObj['StyleAccordion']);
    }
    return StyleAccordion;
}(Style));
var ComponentTable = (function (_super) {
    __extends(ComponentTable, _super);
    function ComponentTable(jsonStr) {
        _super.call(this, ComponentType.ComponentTable);
        this.render = function (appendToObject) {
            var s = this.style;
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
            if (this.header) {
                var theader = document.createElement('thead');
                var headerRow = document.createElement('tr');
                if (s && s.getHeaderColor())
                    headerRow.style.backgroundColor = s.getHeaderColor();
                for (var i = 0; i < this.header.length; i++) {
                    var headerd = document.createElement('th');
                    headerd.style.padding = padTop + 'px ' + padRight + 'px ' + padBottom + 'px ' + padLeft + 'px';
                    headerd.appendChild(document.createTextNode(this.header[i]));
                    headerRow.appendChild(headerd);
                }
                tbl.appendChild(headerRow);
            }
            if (this.content) {
                var tbdy = document.createElement('tbody');
                for (var i = 0; i < this.content.length; i++) {
                    var tr = document.createElement('tr');
                    for (var j = 0; j < this.content[i].length; j++) {
                        var td = document.createElement('td');
                        td.style.padding = padTop + 'px ' + padRight + 'px ' + padBottom + 'px ' + padLeft + 'px';
                        td.appendChild(document.createTextNode(this.content[i][j]));
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
        this.header = json['header'];
        this.content = json['content'];
        if (json['style'])
            this.style = new StyleTable(json['style']);
    }
    return ComponentTable;
}(Component));
var StyleTable = (function (_super) {
    __extends(StyleTable, _super);
    function StyleTable(jsonObj) {
        var _this = this;
        _super.call(this, jsonObj['StyleTable']);
        this.getColumnWidths = function () { return _this.columnWidths; };
        this.getColumnWidthUnit = function () { return _this.columnWidthUnit; };
        this.getBorderWidthPx = function () { return _this.borderWidthPx; };
        this.getHeaderColor = function () { return _this.headerColor; };
        this.getWhitespaceMode = function () { return _this.whitespaceMode; };
        var style = jsonObj['StyleTable'];
        if (style) {
            this.columnWidths = jsonObj['StyleTable']['columnWidths'];
            this.borderWidthPx = jsonObj['StyleTable']['borderWidthPx'];
            this.headerColor = jsonObj['StyleTable']['headerColor'];
            this.columnWidthUnit = jsonObj['StyleTable']['columnWidthUnit'];
            this.whitespaceMode = jsonObj['StyleTable']['whitespaceMode'];
        }
    }
    return StyleTable;
}(Style));
var ComponentText = (function (_super) {
    __extends(ComponentText, _super);
    function ComponentText(jsonStr) {
        var _this = this;
        _super.call(this, ComponentType.ComponentText);
        this.render = function (appendToObject) {
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
        this.text = json['text'];
        if (json['style'])
            this.style = new StyleText(json['style']);
    }
    return ComponentText;
}(Component));
var StyleText = (function (_super) {
    __extends(StyleText, _super);
    function StyleText(jsonObj) {
        var _this = this;
        _super.call(this, jsonObj['StyleText']);
        this.getFont = function () { return _this.font; };
        this.getFontSize = function () { return _this.fontSize; };
        this.getUnderline = function () { return _this.underline; };
        this.getColor = function () { return _this.color; };
        this.getWhitespacePre = function () { return _this.whitespacePre; };
        var style = jsonObj['StyleText'];
        if (style) {
            this.font = style['font'];
            this.fontSize = style['fontSize'];
            this.underline = style['underline'];
            this.color = style['color'];
            this.whitespacePre = style['whitespacePre'];
        }
    }
    return StyleText;
}(Style));
//# sourceMappingURL=dl4j-ui.js.map