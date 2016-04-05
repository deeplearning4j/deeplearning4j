
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

}

class ChartStyle extends Style {

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

//class TableStyle extends Style {
//
//}
//
//class TextStyle extends Style {
//
//
//}