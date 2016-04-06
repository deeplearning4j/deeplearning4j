class StyleChart extends Style {

    protected strokeWidth: number;
    protected pointSize: number;
    protected seriesColors: string[];
    protected axisStrokeWidth: number;
    protected titleStyle: StyleText;

    constructor( jsonObj: any ){
        super(jsonObj['StyleChart']);

        var style: any = jsonObj['StyleChart'];

        if(style){
            this.strokeWidth = style['strokeWidth'];
            this.pointSize = style['pointSize'];
            this.seriesColors = style['seriesColors'];
            if(style['titleStyle']) this.titleStyle = new StyleText(style['titleStyle']);
        }
    }

    getStrokeWidth = () => this.strokeWidth;
    getPointSize = () => this.pointSize;
    getSeriesColors = () => this.seriesColors;

    getSeriesColor = (idx: number) => {
        if(!this.seriesColors || idx < 0 || idx > this.seriesColors.length) return null;
        return this.seriesColors[idx];
    };

    getAxisStrokeWidth = () => this.axisStrokeWidth;
    getTitleStyle = () => this.titleStyle;
}