/// <reference path="../../api/Component.ts" />
/// <reference path="../../typedefs/d3.d.ts" />
/// <reference path="../../util/TSUtils.ts" />


import Ordinal = d3.scale.Ordinal;

abstract class Chart extends Component {

    protected style: StyleChart;

    protected title: string;
    protected suppressAxisHorizontal: boolean;
    protected suppressAxisVertical: boolean;
    protected showLegend: boolean;

    protected setXMin: number;
    protected setXMax: number;
    protected setYMin: number;
    protected setYMax: number;

    protected gridVerticalStrokeWidth: number;
    protected gridHorizontalStrokeWidth: number;

    constructor(componentType: ComponentType, jsonStr: string){
        super(componentType);

        var jsonOrig: any = JSON.parse(jsonStr);
        var json: any = jsonOrig[ComponentType[componentType]];

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

        if(json['style']) this.style = new StyleChart(json['style']);
    }

    getStyle(): StyleChart {
        return this.style;
    }

    protected static appendTitle(svg: any, title: string, margin: Margin, titleStyle: StyleText): void{
        var text = svg.append("text")
            .text(title)
            .attr("x", (margin.widthExMargins / 2))
            .attr("y", 0 - ((margin.top - 30) / 2))
            .attr("text-anchor", "middle");

        if(titleStyle){
            if(titleStyle.getFont()) text.attr("font-family",titleStyle.getFont);
            if(titleStyle.getFontSize() != null) text.attr("font-size",titleStyle.getFontSize()+"pt");
            if(titleStyle.getUnderline() != null) text.style("text-decoration", "underline");
            if(titleStyle.getColor()) text.style("fill",titleStyle.getColor);
            else text.style("fill",ChartConstants.DEFAULT_TITLE_COLOR);
        } else {
            text.style("text-decoration", "underline");
            text.style("fill",ChartConstants.DEFAULT_TITLE_COLOR);
        }
    }
}