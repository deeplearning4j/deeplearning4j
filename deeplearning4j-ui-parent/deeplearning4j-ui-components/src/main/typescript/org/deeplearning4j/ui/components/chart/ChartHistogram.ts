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

        var json = JSON.parse(jsonStr)[ComponentType[ComponentType.ChartHistogram]];


        this.lowerBounds = json['lowerBounds'];
        this.upperBounds = json['upperBounds'];
        this.yValues = json['yvalues'];
    }


    render = (appendToObject: JQuery) => {
        var s: ChartStyle = this.getStyle();
        var margin: Margin = Style.getMargins(s);

        // Set the ranges
        var xScale: d3.scale.Linear<number,number> = d3.scale.linear().range([0, margin.widthExMargins]);
        var yScale: d3.scale.Linear<number,number> = d3.scale.linear().range([margin.heightExMargins, 0]);

        // Define the axes
        var xAxis: any = d3.svg.axis().scale(xScale)
            .orient("bottom").ticks(5);
        if(this.gridVerticalStrokeWidth && this.gridVerticalStrokeWidth > 0){
            xAxis.innerTickSize(-margin.heightExMargins);     //used as grid line
        }


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

        // Define the line
        var valueline = d3.svg.line()
            .x(function (d: any) {
                return xScale(d.xPos);
            })
            .y(function (d: any) {
                return yScale(d.yPos);
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


        svg.selectAll(".bin")
            .data(data)
            .enter().append("rect")
            .attr("class", "bin")
            .attr("x", function(d: any) { return xAxis(d.offset); })
            .attr("width", function(d: any) { return xAxis(xMin + d.width) - 1; })
            .attr("y", function(d: any) { return yAxis(d.height); })
            .attr("height", function(d: any) { return margin.heightExMargins - yAxis(d.height); });

        // Add the X Axis
        var xAxisNode = svg.append("g")
            .attr("class", "x axis")
            .attr("transform", "translate(0," + margin.heightExMargins + ")")
            .style("stroke","#000")
            .style("stroke-width", (s != null && s.getAxisStrokeWidth() != null ? s.getAxisStrokeWidth() : ChartConstants.DEFAULT_AXIS_STROKE_WIDTH))
            .style("fill","none")
            .call(xAxis);

        if(this.gridVerticalStrokeWidth != null) xAxisNode.selectAll('.axis line').style({'stroke-width': this.gridVerticalStrokeWidth});

        // Add the Y Axis
        var yAxisNode = svg.append("g")
            .attr("class", "y axis")
            .style("stroke","#000")
            .style("stroke-width", (s != null && s.getAxisStrokeWidth() != null ? s.getAxisStrokeWidth() : ChartConstants.DEFAULT_AXIS_STROKE_WIDTH))
            .style("fill","none")
            .call(yAxis);

        if(this.gridHorizontalStrokeWidth != null) yAxisNode.selectAll('.axis line').style({'stroke-width': this.gridHorizontalStrokeWidth});

        //Add title (if present)
        if (this.title) {
            svg.append("text")
                .attr("x", (margin.widthExMargins / 2))
                .attr("y", 0 - ((margin.top - 30) / 2))
                .attr("text-anchor", "middle")
                .style("font-size", "13px")
                .style("text-decoration", "underline")
                .text(this.title);
        }
    }
}