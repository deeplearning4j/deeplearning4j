/// <reference path="../../api/Component.ts" />
/// <reference path="../../api/Renderable.ts" />
/// <reference path="../../api/Margin.ts" />
/// <reference path="../../util/TSUtils.ts" />
/// <reference path="../../typedefs/jquery.d.ts" />
/// <reference path="../../typedefs/jqueryui.d.ts" />

class DecoratorAccordion extends Component implements Renderable {

    private style: StyleAccordion;
    private title: string;
    private defaultCollapsed: boolean;
    private innerComponents: Renderable[];

    constructor(jsonStr: string){
        super(ComponentType.DecoratorAccordion);

        var json = JSON.parse(jsonStr)[ComponentType[ComponentType.DecoratorAccordion]];

        this.title = json['title'];
        this.defaultCollapsed = json['defaultCollapsed'];

        var innerCs: any[] = json['innerComponents'];

        if(innerCs){
            //this.innerComponents = new Renderable[innerCs.length];
            //this.innerComponents = new Array<Renderable>(innerCs.length);
            this.innerComponents = [];
            for( var i=0; i<innerCs.length; i++ ){
                var asStr: string = JSON.stringify(innerCs[i]);
                this.innerComponents.push(Component.getComponent(asStr));
            }
        }

        if(json['style']) this.style = new StyleAccordion(json['style']);
    }

    render = (appendToObject: JQuery) => {

        var s:StyleAccordion = this.style;

        //var elementdiv: HTMLDivElement = document.createElement("div");
        var outerDiv: JQuery = $('<div></div>');
        outerDiv.uniqueId();

        var titleDiv: JQuery;
        if(this.title) titleDiv = $('<div>' + this.title + '</div>');
        else titleDiv = $('<div></div>');
        titleDiv.uniqueId();
        outerDiv.append(titleDiv);

        var innerDiv: JQuery = $('<div></div>');
        innerDiv.uniqueId();
        outerDiv.append(innerDiv);

        //Add the inner components:
        if (this.innerComponents) {
            for (var i = 0; i < this.innerComponents.length; i++) {
                //this.innerComponents[i].render(outerDiv);
                this.innerComponents[i].render(innerDiv);
            }
        }

        appendToObject.append(outerDiv);

        outerDiv.accordion({collapsible: true, heightStyle: "content"});
        //$(function(){outerDiv.accordion({collapsible: true, heightStyle: "content"})});

        //elementdiv.accordion();
        //$(function(){elementdiv.accordion()});

    }


}