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
    ComponentType[ComponentType["DecoratorAccordion"] = 8] = "DecoratorAccordion";
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
        var key = Object.keys(json)[0];
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
        var json = jsonOrig[ComponentType[componentType]];
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
        var _this = this;
        _super.call(this, ComponentType.ChartHistogram, jsonStr);
        this.render = function (appendToObject) {
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
        var json = JSON.parse(jsonStr)[ComponentType[ComponentType.ChartHistogram]];
        this.lowerBounds = json['lowerBounds'];
        this.upperBounds = json['upperBounds'];
        this.yValues = json['yvalues'];
    }
    return ChartHistogram;
}(Chart));
var ChartLine = (function (_super) {
    __extends(ChartLine, _super);
    function ChartLine(jsonStr) {
        var _this = this;
        _super.call(this, ComponentType.ChartLine, jsonStr);
        this.render = function (appendToObject) {
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
        var json = JSON.parse(jsonStr)[ComponentType[ComponentType.ChartLine]];
        this.xData = json['x'];
        this.yData = json['y'];
        this.seriesNames = json['seriesNames'];
    }
    return ChartLine;
}(Chart));
var ChartScatter = (function (_super) {
    __extends(ChartScatter, _super);
    function ChartScatter(jsonStr) {
        var _this = this;
        _super.call(this, ComponentType.ChartScatter, jsonStr);
        this.render = function (appendToObject) {
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
        var json = JSON.parse(jsonStr)[ComponentType[ComponentType.ChartScatter]];
        this.xData = json['x'];
        this.yData = json['y'];
        this.seriesNames = json['seriesNames'];
    }
    return ChartScatter;
}(Chart));
var ChartStackedArea = (function (_super) {
    __extends(ChartStackedArea, _super);
    function ChartStackedArea(jsonStr) {
        var _this = this;
        _super.call(this, ComponentType.ChartStackedArea, jsonStr);
        this.render = function (appendToObject) {
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
            browser.append("path")
                .attr("class", "area")
                .attr("data-legend", function (d) { return d.name; })
                .attr("d", function (d) {
                return area(d.values);
            })
                .style("fill", function (d) {
                return color(d.name);
            })
                .style({ "stroke-width": "0px" });
            browser.append("text")
                .datum(function (d) {
                return { name: d.name, value: d.values[d.values.length - 1] };
            })
                .attr("transform", function (d) {
                return "translate(" + xScale(d.value.xValue) + "," + yScale(d.value.y0 + d.value.y / 2) + ")";
            })
                .attr("x", -6)
                .attr("dy", ".35em")
                .text(function (d) {
                return d.name;
            });
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
        };
        var json = JSON.parse(jsonStr)[ComponentType[ComponentType.ChartStackedArea]];
        this.xData = json['x'];
        this.yData = json['y'];
        this.labels = json['labels'];
    }
    return ChartStackedArea;
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
            if (!_this.seriesColors || idx < 0 || idx > _this.seriesColors.length)
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
        var _this = this;
        _super.call(this, ComponentType.ComponentDiv);
        this.render = function (appendToObject) {
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
            if (_this.components) {
                for (var i = 0; i < _this.components.length; i++) {
                    _this.components[i].render(newDiv);
                }
            }
            appendToObject.append(newDiv);
        };
        var json = JSON.parse(jsonStr)[ComponentType[ComponentType.ComponentDiv]];
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
        var _this = this;
        _super.call(this, ComponentType.DecoratorAccordion);
        this.render = function (appendToObject) {
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
        var json = JSON.parse(jsonStr)[ComponentType[ComponentType.DecoratorAccordion]];
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
        var _this = this;
        _super.call(this, ComponentType.ComponentTable);
        this.render = function (appendToObject) {
            var s = _this.style;
            var margin = Style.getMargins(s);
            var tbl = document.createElement('table');
            tbl.style.width = '100%';
            tbl.style.height = '100%';
            if (s && s.getBorderWidthPx() != null)
                tbl.setAttribute('border', String(s.getBorderWidthPx()));
            if (s && s.getBackgroundColor())
                tbl.style.backgroundColor = s.getBackgroundColor();
            if (s && s.getColumnWidths()) {
                var colWidths = s.getColumnWidths();
                for (var i = 0; i < colWidths.length; i++) {
                    var col = document.createElement('col');
                    col.setAttribute('width', colWidths[i] + '%');
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
        var json = JSON.parse(jsonStr)[ComponentType[ComponentType.ComponentTable]];
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
        this.getBorderWidthPx = function () { return _this.borderWidthPx; };
        this.getHeaderColor = function () { return _this.headerColor; };
        var style = jsonObj['StyleTable'];
        if (style) {
            this.columnWidths = jsonObj['StyleTable']['columnWidths'];
            this.borderWidthPx = jsonObj['StyleTable']['borderWidthPx'];
            this.headerColor = jsonObj['StyleTable']['headerColor'];
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
                newSpan.appendChild(textNode);
                appendToObject.append(newSpan);
            }
            else {
                var newSpan = document.createElement('span');
                newSpan.appendChild(textNode);
                appendToObject.append(newSpan);
            }
        };
        var json = JSON.parse(jsonStr)[ComponentType[ComponentType.ComponentText]];
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
        var style = jsonObj['StyleText'];
        if (style) {
            this.font = style['font'];
            this.fontSize = style['fontSize'];
            this.underline = style['underline'];
            this.color = style['color'];
        }
    }
    return StyleText;
}(Style));
//# sourceMappingURL=dl4j-ui.js.map