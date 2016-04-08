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

abstract class Style {

    private width: number;
    private height: number;
    private widthUnit: string;
    private heightUnit: string;

    private marginTop: number;
    private marginBottom: number;
    private marginLeft: number;
    private marginRight: number;

    private backgroundColor: string;

    constructor( jsonObj: any){
        this.width = jsonObj['width'];
        this.height = jsonObj['height'];
        this.widthUnit = TSUtils.normalizeLengthUnit(jsonObj['widthUnit']);
        this.heightUnit = TSUtils.normalizeLengthUnit(jsonObj['heightUnit']);
        this.marginTop = jsonObj['marginTop'];
        this.marginBottom = jsonObj['marginBottom'];
        this.marginLeft = jsonObj['marginLeft'];
        this.marginRight = jsonObj['marginRight'];
        this.backgroundColor = jsonObj['backgroundColor'];
    }

    getWidth = () => this.width;
    getHeight = () => this.height;
    getWidthUnit = () => this.widthUnit;
    getHeightUnit = () => this.heightUnit;
    getMarginTop = () => this.marginTop;
    getMarginBottom = () => this.marginBottom;
    getMarginLeft = () => this.marginLeft;
    getMarginRight = () => this.marginRight;
    getBackgroundColor = () => this.backgroundColor;


    static getMargins(s: Style): Margin{
        var mTop: number = (s ? s.getMarginTop() : 0);
        var mBottom: number = (s ? s.getMarginBottom() : 0);
        var mLeft: number = (s ? s.getMarginLeft() : 0);
        var mRight: number = (s ? s.getMarginRight() : 0);

        // Set the dimensions of the canvas / graph
        return {top: mTop,
            right: mRight,
            bottom: mBottom,
            left: mLeft,
            widthExMargins: s.getWidth() - mLeft - mRight,
            heightExMargins: s.getHeight() - mTop - mBottom};
    }
}