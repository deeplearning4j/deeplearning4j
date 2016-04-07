
class ComponentDiv extends Component implements Renderable {

    private style: StyleDiv;
    private components: Renderable[];

    constructor(jsonStr: string){
        super(ComponentType.ComponentDiv);

        var json = JSON.parse(jsonStr)[ComponentType[ComponentType.ComponentDiv]];

        var components: any[] = json['components'];

        if(components){
            this.components = [];
            for( var i=0; i<components.length; i++ ){
                var asStr: string = JSON.stringify(components[i]);
                this.components.push(Component.getComponent(asStr));
            }
        }

        if(json['style']) this.style = new StyleDiv(json['style']);


    }

    render = (appendToObject: JQuery) => {

        var newDiv: JQuery = $('<div></div>');
        newDiv.uniqueId();

        if(this.style){

            if(this.style.getWidth()){
                var unit: string = this.style.getWidthUnit();
                newDiv.width(this.style.getWidth() + (unit ? unit : ""));
            }
            if(this.style.getHeight()){
                var unit: string = this.style.getHeightUnit();
                newDiv.height(this.style.getHeight() + (unit ? unit : ""));
            }
            if(this.style.getBackgroundColor()) newDiv.css("background-color",this.style.getBackgroundColor());
            if(this.style.getFloatValue()) newDiv.css("float", this.style.getFloatValue());
        }

        //now, add the sub-components:
        if(this.components){
            for( var i=0; i<this.components.length; i++ ){
                this.components[i].render(newDiv);
            }
        }

        appendToObject.append(newDiv);
    }

}