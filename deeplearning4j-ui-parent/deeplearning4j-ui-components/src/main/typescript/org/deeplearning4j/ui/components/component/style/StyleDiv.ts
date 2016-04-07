
class StyleDiv extends Style {

    protected floatValue: string;

    constructor( jsonObj: any){
        super(jsonObj['StyleDiv']);

        if(jsonObj && jsonObj['StyleDiv']) this.floatValue = jsonObj['StyleDiv']['floatValue'];

    }

    getFloatValue = () => this.floatValue;


}