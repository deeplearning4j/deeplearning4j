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

class StyleTable extends Style {

    private columnWidths: number[];
    private columnWidthUnit: string;
    private borderWidthPx: number;
    private headerColor: string;
    private whitespaceMode: string;

    constructor( jsonObj: any ){
        super(jsonObj['StyleTable']);

        var style: any = jsonObj['StyleTable'];
        if(style){
            this.columnWidths = jsonObj['StyleTable']['columnWidths'];
            this.borderWidthPx = jsonObj['StyleTable']['borderWidthPx'];
            this.headerColor = jsonObj['StyleTable']['headerColor'];
            this.columnWidthUnit = jsonObj['StyleTable']['columnWidthUnit'];
            this.whitespaceMode = jsonObj['StyleTable']['whitespaceMode'];
        }
    }

    getColumnWidths = () => this.columnWidths;
    getColumnWidthUnit = () => this.columnWidthUnit;
    getBorderWidthPx = () => this.borderWidthPx;
    getHeaderColor = () => this.headerColor;
    getWhitespaceMode = () => this.whitespaceMode;
}