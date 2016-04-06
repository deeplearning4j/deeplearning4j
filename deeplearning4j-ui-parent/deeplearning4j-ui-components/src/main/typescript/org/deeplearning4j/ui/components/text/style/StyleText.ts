class StyleText extends Style {

    private font: string;
    private fontSize: number;
    private underline: boolean;
    private color: string;

    constructor( jsonObj: any){
        super(jsonObj['StyleText']);

        var style: any = jsonObj['StyleText'];
        if(style){
            this.font = style['font'];
            this.fontSize = style['fontSize'];
            this.underline = style['underline'];
            this.color = style['color'];
        }
    }

    getFont = () => this.font;
    getFontSize = () => this.fontSize;
    getUnderline = () => this.underline;
    getColor = () => this.color;
}