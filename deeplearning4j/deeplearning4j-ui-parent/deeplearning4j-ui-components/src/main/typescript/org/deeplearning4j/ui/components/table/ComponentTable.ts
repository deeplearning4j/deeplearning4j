/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

/// <reference path="../../api/Component.ts" />
/// <reference path="../../api/Margin.ts" />
/// <reference path="../../util/TSUtils.ts" />

class ComponentTable extends Component implements Renderable {

    private header: string[];
    private content: string[][];
    private style: StyleTable;


    constructor(jsonStr: string){
        super(ComponentType.ComponentTable);

        var json = JSON.parse(jsonStr);
        if(!json["componentType"]) json = json[ComponentType[ComponentType.ComponentTable]];

        this.header = json['header'];
        this.content = json['content'];
        if(json['style']) this.style = new StyleTable(json['style']);
    }

    render = (appendToObject: JQuery) => {

        var s: StyleTable = this.style;
        var margin: Margin = Style.getMargins(s);

        var tbl = document.createElement('table');
        //TODO allow setting of table width/height
        tbl.style.width = '100%';
        if(s && s.getBorderWidthPx() != null ) tbl.setAttribute('border', String(s.getBorderWidthPx()));
        if(s && s.getBackgroundColor()) tbl.style.backgroundColor = s.getBackgroundColor();
        if(s && s.getWhitespaceMode()) tbl.style.whiteSpace = s.getWhitespaceMode();

        if (s && s.getColumnWidths()) {
            //TODO allow other than percentage
            var colWidths: number[] = s.getColumnWidths();
            var unit: string = TSUtils.normalizeLengthUnit(s.getColumnWidthUnit());
            for (var i = 0; i < colWidths.length; i++) {
                var col = document.createElement('col');
                col.setAttribute('width', colWidths[i] + unit);
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