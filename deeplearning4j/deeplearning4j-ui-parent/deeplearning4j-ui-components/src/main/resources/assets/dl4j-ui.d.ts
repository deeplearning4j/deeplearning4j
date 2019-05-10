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

/// <reference path="../../typescript/org/deeplearning4j/ui/typedefs/jquery.d.ts" />
/// <reference path="../../typescript/org/deeplearning4j/ui/typedefs/d3.d.ts" />
/// <reference path="../../typescript/org/deeplearning4j/ui/typedefs/jqueryui.d.ts" />
declare abstract class Style {
    private width;
    private height;
    private widthUnit;
    private heightUnit;
    private marginTop;
    private marginBottom;
    private marginLeft;
    private marginRight;
    private backgroundColor;
    constructor(jsonObj: any);
    getWidth: () => number;
    getHeight: () => number;
    getWidthUnit: () => string;
    getHeightUnit: () => string;
    getMarginTop: () => number;
    getMarginBottom: () => number;
    getMarginLeft: () => number;
    getMarginRight: () => number;
    getBackgroundColor: () => string;
    static getMargins(s: Style): Margin;
}
declare enum ComponentType {
    ComponentText = 0,
    ComponentTable = 1,
    ComponentDiv = 2,
    ChartHistogram = 3,
    ChartHorizontalBar = 4,
    ChartLine = 5,
    ChartScatter = 6,
    ChartStackedArea = 7,
    ChartTimeline = 8,
    DecoratorAccordion = 9,
}
declare abstract class Component {
    private componentType;
    constructor(componentType: ComponentType);
    getComponentType(): ComponentType;
    static getComponent(jsonStr: string): Renderable;
}
declare class ChartConstants {
    static DEFAULT_CHART_STROKE_WIDTH: number;
    static DEFAULT_CHART_POINT_SIZE: number;
    static DEFAULT_AXIS_STROKE_WIDTH: number;
    static DEFAULT_TITLE_COLOR: string;
}
interface Margin {
    top: number;
    right: number;
    bottom: number;
    left: number;
    widthExMargins: number;
    heightExMargins: number;
}
interface Renderable {
    render: (addToObject: JQuery) => void;
}
declare class TSUtils {
    static max(input: number[][]): number;
    static min(input: number[][]): number;
    static normalizeLengthUnit(input: string): string;
}
declare abstract class Chart extends Component {
    protected style: StyleChart;
    protected title: string;
    protected suppressAxisHorizontal: boolean;
    protected suppressAxisVertical: boolean;
    protected showLegend: boolean;
    protected setXMin: number;
    protected setXMax: number;
    protected setYMin: number;
    protected setYMax: number;
    protected gridVerticalStrokeWidth: number;
    protected gridHorizontalStrokeWidth: number;
    constructor(componentType: ComponentType, jsonStr: string);
    getStyle(): StyleChart;
    protected static appendTitle(svg: any, title: string, margin: Margin, titleStyle: StyleText): void;
}
declare class ChartHistogram extends Chart implements Renderable {
    private lowerBounds;
    private upperBounds;
    private yValues;
    constructor(jsonStr: string);
    render: (appendToObject: JQuery) => void;
}
declare class ChartLine extends Chart implements Renderable {
    private xData;
    private yData;
    private seriesNames;
    constructor(jsonStr: string);
    render: (appendToObject: JQuery) => void;
}
declare class ChartScatter extends Chart implements Renderable {
    private xData;
    private yData;
    private seriesNames;
    constructor(jsonStr: string);
    render: (appendToObject: JQuery) => void;
}
declare class Legend {
    private static offsetX;
    private static offsetY;
    private static padding;
    private static separation;
    private static boxSize;
    private static fillColor;
    private static legendOpacity;
    private static borderStrokeColor;
    static legendFn: (g: any) => void;
}
declare class ChartStackedArea extends Chart implements Renderable {
    private xData;
    private yData;
    private labels;
    constructor(jsonStr: string);
    render: (appendToObject: JQuery) => void;
}
declare class ChartTimeline extends Chart implements Renderable {
    private laneNames;
    private laneData;
    private lanes;
    private itemData;
    private mainView;
    private miniView;
    private brush;
    private x;
    private x1;
    private xTimeAxis;
    private y1;
    private y2;
    private itemRects;
    private rect;
    private static MINI_LANE_HEIGHT_PX;
    private static ENTRY_LANE_HEIGHT_OFFSET_FRACTION;
    private static ENTRY_LANE_HEIGHT_TOTAL_FRACTION;
    private static MILLISEC_PER_MINUTE;
    private static MILLISEC_PER_HOUR;
    private static MILLISEC_PER_DAY;
    private static MILLISEC_PER_WEEK;
    private static DEFAULT_COLOR;
    constructor(jsonStr: string);
    render: (appendToObject: JQuery) => void;
    renderChart: () => void;
    moveBrush: () => void;
    getMiniViewPaths: (items: any) => any[];
}
declare class StyleChart extends Style {
    protected strokeWidth: number;
    protected pointSize: number;
    protected seriesColors: string[];
    protected axisStrokeWidth: number;
    protected titleStyle: StyleText;
    constructor(jsonObj: any);
    getStrokeWidth: () => number;
    getPointSize: () => number;
    getSeriesColors: () => string[];
    getSeriesColor: (idx: number) => string;
    getAxisStrokeWidth: () => number;
    getTitleStyle: () => StyleText;
}
declare class ComponentDiv extends Component implements Renderable {
    private style;
    private components;
    constructor(jsonStr: string);
    render: (appendToObject: JQuery) => void;
}
declare class StyleDiv extends Style {
    protected floatValue: string;
    constructor(jsonObj: any);
    getFloatValue: () => string;
}
declare class DecoratorAccordion extends Component implements Renderable {
    private style;
    private title;
    private defaultCollapsed;
    private innerComponents;
    constructor(jsonStr: string);
    render: (appendToObject: JQuery) => void;
}
declare class StyleAccordion extends Style {
    constructor(jsonObj: any);
}
declare class ComponentTable extends Component implements Renderable {
    private header;
    private content;
    private style;
    constructor(jsonStr: string);
    render: (appendToObject: JQuery) => void;
}
declare class StyleTable extends Style {
    private columnWidths;
    private columnWidthUnit;
    private borderWidthPx;
    private headerColor;
    private whitespaceMode;
    constructor(jsonObj: any);
    getColumnWidths: () => number[];
    getColumnWidthUnit: () => string;
    getBorderWidthPx: () => number;
    getHeaderColor: () => string;
    getWhitespaceMode: () => string;
}
declare class ComponentText extends Component implements Renderable {
    private text;
    private style;
    constructor(jsonStr: string);
    render: (appendToObject: JQuery) => void;
}
declare class StyleText extends Style {
    private font;
    private fontSize;
    private underline;
    private color;
    private whitespacePre;
    constructor(jsonObj: any);
    getFont: () => string;
    getFontSize: () => number;
    getUnderline: () => boolean;
    getColor: () => string;
    getWhitespacePre: () => boolean;
}
