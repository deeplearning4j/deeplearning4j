/// <reference path="../../api/Component.ts" />
/// <reference path="../../api/Renderable.ts" />
/// <reference path="../../typedefs/d3.d.ts" />
/// <reference path="../../util/TSUtils.ts" />
/// <reference path="Chart.ts" />


class ChartStackedArea extends Chart implements Renderable {
    private xData: number[];
    private yData: number[][];
    private labels: string[];

    constructor(jsonStr: string){
        super(ComponentType.ChartStackedArea, jsonStr);

        var json = JSON.parse(jsonStr)[ComponentType[ComponentType.ChartStackedArea]];


        this.xData = json['x'];
        this.yData = json['y'];
        this.labels = json['labels'];
    }


    render = (appendToObject: JQuery) => {

        var nSeries: number = (!this.xData ? 0 : this.xData.length);
        var s: StyleChart = this.getStyle();
        var margin: Margin = Style.getMargins(s);

        // Set the ranges
        var xScale: d3.scale.Linear<number,number> = d3.scale.linear().range([0, margin.widthExMargins]);
        var yScale: d3.scale.Linear<number,number> = d3.scale.linear().range([margin.heightExMargins, 0]);

        // Define the axes
        var xAxis = d3.svg.axis().scale(xScale)
            .orient("bottom").ticks(5);
        if(this.gridVerticalStrokeWidth != null && this.gridVerticalStrokeWidth > 0){
            xAxis.innerTickSize(-margin.heightExMargins);     //used as grid line
        }


        var yAxis = d3.svg.axis().scale(yScale)
            .orient("left").ticks(5);
        if(this.gridHorizontalStrokeWidth != null && this.gridHorizontalStrokeWidth > 0){
            yAxis.innerTickSize(-margin.widthExMargins);      //used as grid line
        }

        if(this.suppressAxisHorizontal === true) xAxis.tickValues([]);

        if(this.suppressAxisVertical === true) yAxis.tickValues([]);

        var data: any[] = [];
        for(var i=0; i<this.xData.length; i++ ){
            var obj = {};
            for( var j=0; j<this.labels.length; j++ ){
                obj[this.labels[j]] = this.yData[j][i];
                obj['xValue'] = this.xData[i];
            }
            data.push(obj);
        }

        var area = d3.svg.area()
            .x(function(d: any) { return xScale(d.xValue); })
            .y0(function(d: any) { return yScale(d.y0); })
            .y1(function(d: any) { return yScale(d.y0 + d.y); });

        var stack = d3.layout.stack()
            .values(function(d: any) { return d.values; });

        var svg = d3.select("#" + appendToObject.attr("id")).append("svg")
            .attr("width", margin.widthExMargins + margin.left + margin.right)
            .attr("height", margin.heightExMargins + margin.top + margin.bottom)
            .append("g")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        var color: any = d3.scale.category20();
        color.domain(d3.keys(data[0]).filter(function (key) {
            return key !== "xValue";
        }));

        var browsers = stack(color.domain().map(function (name) {
            return {
                name: name,
                values: data.map(function (d) {
                    return {xValue: d.xValue, y: d[name] * 1};
                })
            };
        }));

        // Find the value of the day with highest total value
        var maxX = d3.max(data, function (d) {
            var vals = d3.keys(d).map(function (key) {
                return key !== "xValue" ? d[key] : 0
            });
            return d3.sum(vals);
        });

        // Set domains for axes
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
            .attr("data-legend",function(d: any) { return d.name})
            .attr("d", function (d: any) {
                return area(d.values);
            })
            .style("fill", function (d: any) {
                return color(d.name);
            })
            .style({"stroke-width": "0px"});

        //This appends the text labels to the right of the chart
        browser.append("text")
                .datum(function (d: any) {
                    return {name: d.name, value: d.values[d.values.length - 1]};
                })
                .attr("transform", function (d) {
                    return "translate(" + xScale(d.value.xValue) + "," + yScale(d.value.y0 + d.value.y / 2) + ")";
                })
                .attr("x", -6)
                .attr("dy", ".35em")
                .text(function (d) {
                    return d.name;
                });

        //Add the x axis:
        var xAxisNode = svg.append("g")
            .attr("class", "x axis")
            .style("stroke","#000")
            .style("stroke-width", (s != null && s.getAxisStrokeWidth() != null ? s.getAxisStrokeWidth() : ChartConstants.DEFAULT_AXIS_STROKE_WIDTH))
            .style("fill","none")
            .attr("transform", "translate(0," + margin.heightExMargins + ")")
            .call(xAxis);
        xAxisNode.selectAll('text').style("stroke-width",0).style("fill","#000000");

        //Add the y axis
        var yAxisNode = svg.append("g")
            .attr("class", "y axis")
            .style("stroke","#000")
            .style("stroke-width", (s != null && s.getAxisStrokeWidth() != null ? s.getAxisStrokeWidth() : ChartConstants.DEFAULT_AXIS_STROKE_WIDTH))
            .style("fill","none")
            .call(yAxis);
        yAxisNode.selectAll('text').style("stroke-width",0).style("fill","#000000");

        //Add title (if present)
        if (this.title) {
            var titleStyle: StyleText;
            if(this.style) titleStyle = this.style.getTitleStyle();
            Chart.appendTitle(svg, this.title, margin, titleStyle);
        }
    }
}