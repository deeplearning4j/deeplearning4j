/// <reference path="../../api/Component.ts" />
/// <reference path="../../api/Margin.ts" />
/// <reference path="../../util/TSUtils.ts" />

class ComponentTable extends Component implements Renderable {

    private header: string[];
    private content: string[][];
    private style: StyleTable;


    constructor(jsonStr: string){
        super(ComponentType.ComponentTable);

        var json = JSON.parse(jsonStr)[ComponentType[ComponentType.ComponentTable]];

        this.header = json['header'];
        this.content = json['content'];
        if(json['style']) this.style = new StyleTable(json['style']);
    }

    render = (appendToObject: JQuery) => {

        var s: StyleTable = this.style;
        var margin: Margin = Style.getMargins(s);

        var tbl = document.createElement('table');
        tbl.style.width = '100%';
        tbl.style.height = '100%';
        if(s && s.getBorderWidthPx() != null ) tbl.setAttribute('border', String(s.getBorderWidthPx()));
        if(s && s.getBackgroundColor()) tbl.style.backgroundColor = s.getBackgroundColor();

        if (s && s.getColumnWidths()) {
            //TODO allow other than percentage
            var colWidths: number[] = s.getColumnWidths();
            for (var i = 0; i < colWidths.length; i++) {
                var col = document.createElement('col');
                col.setAttribute('width', colWidths[i] + '%');
                tbl.appendChild(col);
            }
        }

        //TODO: don't hardcode
        var padTop = 1;
        var padRight = 1;
        var padBottom = 1;
        var padLeft = 1;

        if (this.header) {
            var theader = document.createElement('thead');
            var headerRow = document.createElement('tr');

            if(s && s.getHeaderColor()) headerRow.style.backgroundColor = s.getHeaderColor();

            for (var i = 0; i < this.header.length; i++) {
                var headerd = document.createElement('th');
                headerd.style.padding = padTop + 'px ' + padRight + 'px ' + padBottom + 'px ' + padLeft + 'px';
                headerd.appendChild(document.createTextNode(this.header[i]));
                headerRow.appendChild(headerd);
            }
            tbl.appendChild(headerRow);
        }

        //Add content:
        if (this.content) {

            var tbdy = document.createElement('tbody');
            for (var i = 0; i < this.content.length; i++) {
                var tr = document.createElement('tr');

                for (var j = 0; j < this.content[i].length; j++) {
                    var td = document.createElement('td');
                    td.style.padding = padTop + 'px ' + padRight + 'px ' + padBottom + 'px ' + padLeft + 'px';
                    td.appendChild(document.createTextNode(this.content[i][j]));
                    tr.appendChild(td);
                }

                tbdy.appendChild(tr);
            }
            tbl.appendChild(tbdy);
        }

        appendToObject.append(tbl);
    }


}