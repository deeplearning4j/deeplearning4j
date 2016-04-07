
abstract class Style {

    private width: number;
    private height: number;
    private widthUnit: string;
    private heightUnit: string;

    private marginTop: number;
    private marginBottom: number;
    private marginLeft: number;
    private marginRight: number;

    private backgroundColor: string;

    constructor( jsonObj: any){
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

    getWidth = () => this.width;
    getHeight = () => this.height;
    getWidthUnit = () => this.widthUnit;
    getHeightUnit = () => this.heightUnit;
    getMarginTop = () => this.marginTop;
    getMarginBottom = () => this.marginBottom;
    getMarginLeft = () => this.marginLeft;
    getMarginRight = () => this.marginRight;
    getBackgroundColor = () => this.backgroundColor;


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