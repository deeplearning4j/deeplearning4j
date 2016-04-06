class StyleTable extends Style {

    private columnWidths: number[];
    private borderWidthPx: number;
    private headerColor: string;
    private backgroundColor: string;

    constructor( jsonObj: any ){
        super(jsonObj['StyleTable']);

        var style: any = jsonObj['StyleTable'];
        if(style){
            this.columnWidths = jsonObj['StyleTable']['columnWidths'];
            this.borderWidthPx = jsonObj['StyleTable']['borderWidthPx'];
            this.headerColor = jsonObj['StyleTable']['headerColor'];
            this.backgroundColor = jsonObj['StyleTable']['backgroundColor'];
        }
    }

    getColumnWidths = () => this.columnWidths;

    getBorderWidthPx = () => this.borderWidthPx;

    getHeaderColor = () => this.headerColor;

    getBackgroundColor = () => this.backgroundColor;

}