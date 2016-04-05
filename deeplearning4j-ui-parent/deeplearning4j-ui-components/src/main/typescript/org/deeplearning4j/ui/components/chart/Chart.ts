/// <reference path="../../api/Component.ts" />
/// <reference path="../../typedefs/d3.d.ts" />
/// <reference path="../../util/TSUtils.ts" />


import Ordinal = d3.scale.Ordinal;

class ChartConstants {

    static DEFAULT_AXIS_STROKE_WIDTH = 1.0;

}

abstract class Chart extends Component {

    protected style: ChartStyle;

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

        if(json['style']) this.style = new ChartStyle(json['style']);
    }

    getStyle(): ChartStyle {
        return this.style;
    }
}