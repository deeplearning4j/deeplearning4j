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

class ComponentText extends Component implements Renderable {

    private text: string;
    private style: StyleText;

    constructor(jsonStr: string){
        super(ComponentType.ComponentText);
        var json = JSON.parse(jsonStr);
        if(!json["componentType"]) json = json[ComponentType[ComponentType.ComponentText]];

        this.text = json['text'];

        if(json['style']) this.style = new StyleText(json['style']);
    }

    render = (appendToObject: JQuery) => {

        var textNode: Text = document.createTextNode(this.text);
        if(this.style){
            var newSpan: HTMLSpanElement = document.createElement('span');
            if(this.style.getFont()) newSpan.style.font = this.style.getFont();
            if(this.style.getFontSize() != null) newSpan.style.fontSize = this.style.getFontSize() + "pt";
            if(this.style.getUnderline() != null) newSpan.style.textDecoration='underline';
            if(this.style.getColor()) newSpan.style.color = this.style.getColor();
            if(this.style.getMarginTop()) newSpan.style.marginTop = this.style.getMarginTop() + "px";
            if(this.style.getMarginBottom()) newSpan.style.marginBottom = this.style.getMarginBottom() + "px";
            if(this.style.getMarginLeft()) newSpan.style.marginLeft = this.style.getMarginLeft() + "px";
            if(this.style.getMarginRight()) newSpan.style.marginRight = this.style.getMarginRight() + "px";
            if(this.style.getWhitespacePre()) newSpan.style.whiteSpace = 'pre';

            newSpan.appendChild(textNode);
            appendToObject.append(newSpan);
        } else {
            var newSpan: HTMLSpanElement = document.createElement('span');
            //appendToObject.append(textNode);
            newSpan.appendChild(textNode);
            appendToObject.append(newSpan);
        }
    }

}