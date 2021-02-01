/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */


function renderLineChart(/*jquery selector*/ element, label, xDataArray, yDataArray ){
    var toPlot = [];
    for(var i=0; i<xDataArray.length; i++ ){
        toPlot.push([xDataArray[i], yDataArray[i]]);
    }

    element.unbind();

    var yMax = Math.max.apply(Math, yDataArray);
    var yMin = Math.min.apply(Math, yDataArray);
    if(yMin > 0){
        yMin = 0.0;
    }

    var plotOptions = {
        series: {
            lines: {
                show: true,
                lineWidth: 2
            }
        },
        grid: {
            hoverable: true,
            clickable: true,
            tickColor: "#dddddd",
            borderWidth: 0
        },
        yaxis: {min: yMin, max: yMax},
        colors: ["#FA5833","rgba(65,182,240,0.3)","#000000"],
        selection: {
            mode: "x"
        }
    };

    var plotData = [{data: toPlot, label: label}];
    var plot = $.plot(element, plotData, plotOptions);
}

function renderHistogramSingle(/*jquery selector*/ element, label, /*nd4j.graph.UIEvent*/ evt, /*nd4j.graph.UIHistogram*/ h){
    if(evt == null || h == null){
        return;
    }

    //Histogram rendering:

    var data = [];
    var y = h.y();
    if(h.type() === nd4j.graph.UIHistogramType.EQUAL_SPACING){
        var minmaxArr = h.binranges();  //Rank 1, size 2
        var min = scalarFromFlatArrayIdx(minmaxArr, 0);
        var max = scalarFromFlatArrayIdx(minmaxArr, 1);
        var numBins = h.numbins();

        //Render this as a line chart for now. Could do this as a bar chart instead, but this provides precise control...
        var step = (max-min)/numBins;
        for(var i=0; i<numBins; i++ ){
            var lower = min + step * i;
            var upper = lower + step;
            var yValue = scalarFromFlatArrayIdx(y, i);
            data.push([lower,0]);
            data.push([lower,yValue]);
            data.push([upper,yValue]);
            data.push([upper,0]);
        }
    } else if(h.type() === nd4j.graph.UIHistogramType.DISCRETE){
        var binLabelsCount = h.binlabelsLength();
        var lbl = [];
        for(var i=0; i<binLabelsCount; i++ ){
            lbl.push(h.binlabels(i));
        }
        var min = 0;
        var max = 1;
        var numBins = lbl.length;
        //Render this as a line chart for now. Could do this as a bar chart instead, but this provides precise control...
        var step = (max-min)/numBins;
        for(var i=0; i<numBins; i++ ){
            var lower = min + step * i;
            var upper = lower + step;
            var yValue = scalarFromFlatArrayIdx(y, i);
            data.push([lower,0]);
            data.push([lower,yValue]);
            data.push([upper,yValue]);
            data.push([upper,0]);
        }
    } else if(h.type() === nd4j.graph.UIHistogramType.CUSTOM){
        var minmaxArr = h.binranges();  //Rank 2, shape [2,numBins]
        var numBins = h.numbins();

        for(var i=0; i<numBins; i++ ){
            var lower = getScalar(minmaxArr, [0, i]);
            var upper = getScalar(minmaxArr, [1, i]);
            var yValue = scalarFromFlatArrayIdx(y, i);

            data.push([lower,0]);
            data.push([lower,yValue]);
            data.push([upper,yValue]);
            data.push([upper,0]);
        }
    }

    var plotData = [{data: data, label: label, lines: { show: true, fill: true }}];
    $.plot(element, plotData)

}

var sdEventNamesMap = new Map();        //Map<Integer,String>   - key is name index, value is name
var sdPlotsLineChartX = new Map();      //Map<String,nd4j.graph.UIEvent>
var sdPlotsLineChartY = new Map();      //Map<String,Number[]>
var sdPlotsHistogramX = new Map();      //Map<String,nd4j.graph.UIEvent>
var sdPlotsHistogramY = new Map();      //Map<String,nd4j.graph.UIHistogram>
function readAndRenderPlotsData(){

    if (file) {
        var fr = new FileReader();
        var fileData = new Blob([file]);        //TODO Don't load the whole file into memory at once!
        fr.readAsArrayBuffer(fileData);
        fr.onload = function () {
            var arrayBuffer = fr.result;
            var bytes = new Uint8Array(arrayBuffer);
            //console.log(bytes);

            var currentOffset = 0;
            var foundStartEvents = false;
            var numBytes = bytes.length;
            while(currentOffset < numBytes) {

                var lengths = extractHeaders(bytes, currentOffset);
                var headerLength = lengths[0];
                var contentLength = lengths[1];

                //TODO is there a way to do this with views, not slices?
                var headerSlice = bytes.slice(currentOffset + 8, currentOffset + 8 + headerLength);
                var headerBuffer = new flatbuffers.ByteBuffer(headerSlice);
                var header = nd4j.graph.UIStaticInfoRecord.getRootAsUIStaticInfoRecord(headerBuffer);

                currentOffset += 8 + headerLength + contentLength;

                if(header.infoType() == nd4j.graph.UIInfoType.START_EVENTS){
                    foundStartEvents = true;
                    break;
                }
            }

            if(foundStartEvents){
                //"Start events" marker found... we *might* have some data to plot

                sdEventNamesMap = new Map();
                sdPlotsLineChartsX = new Map();
                sdPlotLineChartsY = new Map();

                while(currentOffset < numBytes) {

                    var lengths = extractHeaders(bytes, currentOffset);
                    var headerLength = lengths[0];
                    var contentLength = lengths[1];

                    var headerSlice = bytes.slice(currentOffset + 8, currentOffset + 8 + headerLength);
                    var headerBuffer = new flatbuffers.ByteBuffer(headerSlice);
                    var header = nd4j.graph.UIEvent.getRootAsUIEvent(headerBuffer);

                    //TODO only slice if it's something we want to decode...
                    var contentSlice = bytes.slice(currentOffset + 8 + headerLength, currentOffset + 8 + headerLength + contentLength);
                    var contentBuffer = new flatbuffers.ByteBuffer(contentSlice);

                    var nameId = header.nameIdx();

                    var evtType = header.eventType();
                    if(evtType === nd4j.graph.UIEventType.ADD_NAME){
                        var content = nd4j.graph.UIAddName.getRootAsUIAddName(contentBuffer);
                        console.log("Decoded ADD_NAME event: " + content.name());
                        var name = content.name();
                        var nameIdx = content.nameIdx();
                        sdEventNamesMap.set(nameIdx, name);

                    } else if(evtType === nd4j.graph.UIEventType.SCALAR){
                        var content = nd4j.graph.FlatArray.getRootAsFlatArray(contentBuffer);
                        var name = sdEventNamesMap.get(nameId);
                        var scalar = scalarFromFlatArray(content);
                        var dt = dataTypeToString(content.dtype());
                        // console.log("Decoded SCALAR event: " + scalar + " - " + dt);

                        if(!sdPlotsLineChartX.has(name)){
                            sdPlotsLineChartX.set(name, []);
                            sdPlotsLineChartY.set(name, []);
                        }

                        sdPlotsLineChartX.get(name).push(header);
                        sdPlotsLineChartY.get(name).push(scalar);
                    } else if(evtType === nd4j.graph.UIEventType.HISTOGRAM){
                        var content = nd4j.graph.UIHistogram.getRootAsUIHistogram(contentBuffer);
                        var name = sdEventNamesMap.get(nameId);

                        if(!sdPlotsHistogramX.has(name)){
                            sdPlotsHistogramX.set(name, []);
                            sdPlotsHistogramY.set(name, []);
                        }

                        sdPlotsHistogramX.get(name).push(header);
                        sdPlotsHistogramY.get(name).push(content);
                    }

                    //TODO other types!

                    currentOffset += 8 + headerLength + contentLength;
                }
            }
        };

        renderLineCharts();
        renderHistograms();
    }
}

function renderLineCharts(){
    var contentDiv = $("#samediffcontent");
    //List available charts, histograms, etc:
    var lineChartKeys = Array.from(sdPlotsLineChartX.keys());
    console.log("Line chart keys: " + lineChartKeys);
    var content1 = "<div><b>Scalars Values</b>:\n" + lineChartKeys.join("\n") + "<br><br></div>";
    contentDiv.html(content1);

    for( var i=0; i<lineChartKeys.length; i++ ) {


        var chartName = "sdLineChart_" + i;
        var chartDivTxt = "\n<div id=\"" + chartName + "\" class=\"center\" style=\"height: 300px; max-width:750px\" ></div>";
        contentDiv.append(chartDivTxt);
        var element = $("#" + chartName);
        var label = lineChartKeys[i];
        var x = sdPlotsLineChartX.get(label);       //nd4j.graph.UIEvent
        var y = sdPlotsLineChartY.get(label);

        //Parse to iteration. We'll want to make this customizable eventually (iteration, time, etc)
        var xPlot = [];
        for( var j=0; j<x.length; j++ ){
            var iter = x[j].iteration();
            xPlot.push(iter);
        }

        renderLineChart(element, label, xPlot, y);
    }
}

function renderHistograms(){
    var contentDiv = $("#samediffcontent");
    var content1 = "<br><br><div><b>Histograms</b>: TODO<br><br></div>";
    contentDiv.append(content1);

    var keys = Array.from(sdPlotsHistogramX.keys());
    console.log("Histogram keys: " + keys);

    for( var i=0; i<keys.length; i++ ){
        var chartName = "sdHistogram_" + i;
        var chartDivTxt = "\n<div id=\"" + chartName + "\" class=\"center\" style=\"height: 300px; max-width:750px\" ></div>";
        contentDiv.append(chartDivTxt);
        var element = $("#" + chartName);
        var label = keys[i];
        var x = sdPlotsHistogramX.get(label);       //nd4j.graph.UIEvent
        var hist = sdPlotsHistogramY.get(label);    //nd4j.graph.UIHistogram
        var h = null;
        var evt = null;
        if(hist.length > 0){
            evt = x[x.length-1];
            h = hist[hist.length-1];
        }

        //Parse to iteration. We'll want to make this customizable eventually (iteration, time, etc)
        var xPlot = [];
        for( var j=0; j<x.length; j++ ){
            var iter = x[j].iteration();
            xPlot.push(iter);
        }

        renderHistogramSingle(element, label, evt, h);
    }
}


