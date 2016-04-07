class StyleTable extends Style {

    private columnWidths: number[];
    private borderWidthPx: number;
    private headerColor: string;

    constructor( jsonObj: any ){
        super(jsonObj['StyleTable']);

        var style: any = jsonObj['StyleTable'];
        if(style){
            this.columnWidths = jsonObj['StyleTable']['columnWidths'];
            this.borderWidthPx = jsonObj['StyleTable']['borderWidthPx'];
            this.headerColor = jsonObj['StyleTable']['headerColor'];
        }
    }

    getColumnWidths = () => this.columnWidths;

    getBorderWidthPx = () => this.borderWidthPx;

    getHeaderColor = () => this.headerColor;

}