/// <reference path="../../api/Component.ts" />
/// <reference path="../../api/Margin.ts" />
/// <reference path="../../util/TSUtils.ts" />

class ComponentText extends Component implements Renderable {

    private text: string;
    private style: StyleText;

    constructor(jsonStr: string){
        super(ComponentType.ComponentText);
        var json = JSON.parse(jsonStr)[ComponentType[ComponentType.ComponentText]];

        this.text = json['text'];

        if(json['style']) this.style = new StyleText(json['style']);
    }

    render = (appendToObject: JQuery) => {

        var textNode: Text = document.createTextNode(this.text);
        if(this.style){
            var temp1 = this.style.getFont();
            var temp2 = this.style.getFontSize();
            var temp3 = this.style.getUnderline();

            var newSpan: HTMLSpanElement = document.createElement('span');
            if(this.style.getFont()) newSpan.style.font = this.style.getFont();
            if(this.style.getFontSize() != null) newSpan.style.fontSize = this.style.getFontSize() + "pt";
            if(this.style.getUnderline() != null) newSpan.style.textDecoration='underline';

            newSpan.style.setProperty("font",this.style.getFont());

            newSpan.style.fontSize = String(this.style.getFontSize());

            newSpan.appendChild(textNode);
            appendToObject.append(newSpan);
        } else {
            appendToObject.append(textNode);
        }
    }

}