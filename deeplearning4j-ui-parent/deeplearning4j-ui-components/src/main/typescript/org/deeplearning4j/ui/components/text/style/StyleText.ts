/*
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */
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