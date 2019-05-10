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

class StyleText extends Style {

    private font: string;
    private fontSize: number;
    private underline: boolean;
    private color: string;
    private whitespacePre: boolean;

    constructor( jsonObj: any){
        super(jsonObj['StyleText']);

        var style: any = jsonObj['StyleText'];
        if(style){
            this.font = style['font'];
            this.fontSize = style['fontSize'];
            this.underline = style['underline'];
            this.color = style['color'];
            this.whitespacePre = style['whitespacePre'];
        }
    }

    getFont = () => this.font;
    getFontSize = () => this.fontSize;
    getUnderline = () => this.underline;
    getColor = () => this.color;
    getWhitespacePre = () => this.whitespacePre;
}