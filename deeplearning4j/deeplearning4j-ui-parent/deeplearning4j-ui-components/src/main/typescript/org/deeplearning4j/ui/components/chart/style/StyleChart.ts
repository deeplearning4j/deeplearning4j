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
class StyleChart extends Style {

    protected strokeWidth: number;
    protected pointSize: number;
    protected seriesColors: string[];
    protected axisStrokeWidth: number;
    protected titleStyle: StyleText;

    constructor( jsonObj: any ){
        super(jsonObj['StyleChart']);

        var style: any = jsonObj['StyleChart'];

        if(style){
            this.strokeWidth = style['strokeWidth'];
            this.pointSize = style['pointSize'];
            this.seriesColors = style['seriesColors'];
            if(style['titleStyle']) this.titleStyle = new StyleText(style['titleStyle']);
        }
    }

    getStrokeWidth = () => this.strokeWidth;
    getPointSize = () => this.pointSize;
    getSeriesColors = () => this.seriesColors;

    getSeriesColor = (idx: number) => {
        if(!this.seriesColors || idx < 0 || idx > this.seriesColors.length) return null;
        return this.seriesColors[idx];
    };

    getAxisStrokeWidth = () => this.axisStrokeWidth;
    getTitleStyle = () => this.titleStyle;
}