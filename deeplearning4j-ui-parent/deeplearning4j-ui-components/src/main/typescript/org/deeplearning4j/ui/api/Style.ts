
abstract class Style {

    private width: number;
    private height: number;
    //TODO: width/height units

    private marginTop: number;
    private marginBottom: number;
    private marginLeft: number;
    private marginRight: number;

    constructor( jsonObj: any){
        //var json = JSON.parse(jsonString)['StyleChart'];

        this.width = jsonObj['width'];
        this.height = jsonObj['height'];
        this.marginTop = jsonObj['marginTop'];
        this.marginBottom = jsonObj['marginBottom'];
        this.marginLeft = jsonObj['marginLeft'];
        this.marginRight = jsonObj['marginRight'];
    }

    getWidth = () => this.width;

    getHeight = () => this.height;

    getMarginTop = () => this.marginTop;

    getMarginBottom = () => this.marginBottom;

    getMarginLeft = () => this.marginLeft;

    getMarginRight = () => this.marginRight;


    static getMargins(s: Style): Margin{
        var mTop: number = (s ? s.getMarginTop() : 0);
        var mBottom: number = (s ? s.getMarginBottom() : 0);
        var mLeft: number = (s ? s.getMarginLeft() : 0);
        var mRight: number = (s ? s.getMarginRight() : 0);

        // Set the dimensions of the canvas / graph
        return {top: mTop,
            right: mRight,
            bottom: mBottom,
            left: mLeft,
            widthExMargins: s.getWidth() - mLeft - mRight,
            heightExMargins: s.getHeight() - mTop - mBottom};
    }
}

class StyleChart extends Style {

    protected strokeWidth: number;
    protected seriesColors: string[];
    protected axisStrokeWidth: number;

    constructor( jsonObj: any ){
        super(jsonObj['StyleChart']);

        this.strokeWidth = jsonObj['StyleChart']['strokeWidth'];
        this.seriesColors = jsonObj['StyleChart']['seriesColors'];
    }

    getStrokeWidth = () => this.strokeWidth;

    getSeriesColors = () => this.seriesColors;

    getSeriesColor = (idx: number) => {
        if(!this.seriesColors || idx < 0 || idx > this.seriesColors.length) return null;
        return this.seriesColors[idx];
    };

    getAxisStrokeWidth = () => this.axisStrokeWidth;
}

class StyleTable extends Style {

    private columnWidths: number[];
    private borderWidthPx: number;
    private headerColor: string;
    private backgroundColor: string;

    constructor( jsonObj: any ){
        super(jsonObj['StyleTable']);

        this.columnWidths = jsonObj['StyleTable']['columnWidths'];
        this.borderWidthPx = jsonObj['StyleTable']['borderWidthPx'];
        this.headerColor = jsonObj['StyleTable']['headerColor'];
        this.backgroundColor = jsonObj['StyleTable']['backgroundColor'];
    }

    getColumnWidths = () => this.columnWidths;

    getBorderWidthPx = () => this.borderWidthPx;

    getHeaderColor = () => this.headerColor;

    getBackgroundColor = () => this.backgroundColor;

}
//
//class StyleText extends Style {
//
//
//}


class StyleAccordion extends Style {

    constructor( jsonObj: any){
        super(jsonObj['StyleAccordion']);


    }

}