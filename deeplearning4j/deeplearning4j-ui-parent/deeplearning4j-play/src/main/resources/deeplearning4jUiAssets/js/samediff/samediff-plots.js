/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

var sdEventNamesMap = new Map();        //Map<Integer,String>   - key is name index, value is name
var sdPlotsLineChartX = new Map();      //Map<String,nd4j.graph.UIEvent>
var sdPlotsLineChartY = new Map();      //Map<String,Number[]>
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

                    var evtType = header.eventType();
                    if(evtType === nd4j.graph.UIEventType.ADD_NAME){
                        var content = nd4j.graph.UIAddName.getRootAsUIAddName(contentBuffer);
                        console.log("Decoded ADD_NAME event: " + content.name());
                        var name = content.name();
                        var nameIdx = content.nameIdx();
                        sdEventNamesMap.set(nameIdx, name);

                        sdPlotsLineChartX.set(name, []);
                        sdPlotsLineChartY.set(name, []);

                    } else if(evtType === nd4j.graph.UIEventType.SCALAR){
                        var content = nd4j.graph.FlatArray.getRootAsFlatArray(contentBuffer);
                        var scalar = scalarFromFlatArray(content);
                        var dt = dataTypeToString(content.dtype());
                        console.log("Decoded SCALAR event: " + scalar + " - " + dt);

                        sdPlotsLineChartX.get(name).push(header);
                        sdPlotsLineChartY.get(name).push(scalar);
                    }

                    //TODO other types!

                    currentOffset += 8 + headerLength + contentLength;
                }
            }
        };

        renderLineCharts();
    }
}

function renderLineCharts(){
    var contentDiv = $("#samediffcontent");
    //List available charts, histograms, etc:
    var lineChartKeys = Array.from(sdPlotsLineChartX.keys());
    var content1 = "<div>Scalars Values: " + lineChartKeys + "<br><br></div>";
    contentDiv.html(content1);

    for( var i=0; i<lineChartKeys.length; i++ ) {


        var chartName = "sdLineChart_" + i;
        var chartDivTxt = "\n<div id=\"" + chartName + "\" class=\"center\" style=\"height: 300px; max-width:750px\" ></div>";
        contentDiv.append(chartDivTxt);
        var element = $("#" + chartName);
        var x = [];
        var y = [];
        renderLineChart(element, label, x, y);
    }
}


