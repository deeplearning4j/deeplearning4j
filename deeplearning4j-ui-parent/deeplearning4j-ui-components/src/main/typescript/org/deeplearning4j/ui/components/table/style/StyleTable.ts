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
class StyleTable extends Style {

    private columnWidths: number[];
    private columnWidthUnit: string;
    private borderWidthPx: number;
    private headerColor: string;

    constructor( jsonObj: any ){
        super(jsonObj['StyleTable']);

        var style: any = jsonObj['StyleTable'];
        if(style){
            this.columnWidths = jsonObj['StyleTable']['columnWidths'];
            this.borderWidthPx = jsonObj['StyleTable']['borderWidthPx'];
            this.headerColor = jsonObj['StyleTable']['headerColor'];
            this.columnWidthUnit = jsonObj['StyleTable']['columnWidthUnit'];
        }
    }

    getColumnWidths = () => this.columnWidths;
    getColumnWidthUnit = () => this.columnWidthUnit;
    getBorderWidthPx = () => this.borderWidthPx;
    getHeaderColor = () => this.headerColor;

}