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

        var json = JSON.parse(jsonStr);
        if(!json["componentType"]) json = json[ComponentType[ComponentType.DecoratorAccordion]];

        this.title = json['title'];
        this.defaultCollapsed = json['defaultCollapsed'];

        var innerCs: any[] = json['innerComponents'];

        if(innerCs){
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

        if(this.defaultCollapsed) outerDiv.accordion({collapsible: true, heightStyle: "content", active: false});
        else outerDiv.accordion({collapsible: true, heightStyle: "content"});




        //$(function(){outerDiv.accordion({collapsible: true, heightStyle: "content"})});

        //elementdiv.accordion();
        //$(function(){elementdiv.accordion()});
    }


}