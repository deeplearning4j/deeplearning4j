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

class ComponentDiv extends Component implements Renderable {

    private style: StyleDiv;
    private components: Renderable[];

    constructor(jsonStr: string){
        super(ComponentType.ComponentDiv);

        var json = JSON.parse(jsonStr);
        if(!json["componentType"]) json = json[ComponentType[ComponentType.ComponentDiv]];

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

        //Adding the div before adding the sub-components seems to be important for charts/d3 to work properly
        appendToObject.append(newDiv);

        //now, add the sub-components:
        if(this.components){
            for( var i=0; i<this.components.length; i++ ){
                this.components[i].render(newDiv);
            }
        }
    }

}