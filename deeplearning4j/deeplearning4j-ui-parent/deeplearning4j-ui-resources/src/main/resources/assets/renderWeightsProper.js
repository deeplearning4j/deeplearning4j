/*
    This is going to be proper version of WeightsRender.js
    TODO list:
    1. Initialize charts once, and just update them with new data <-- DONE
    2. Add 3D surface as overall view for gradients and weights
*/

    var fdata;

    var modelSelector = new Array();
    var magnitudesSelector = new Array();

    // current visible chart, all other charts will be hidden
    var visibleModel = "";
    var visibleGradient = "";
    var visibleMagnitude = "";


    // all charts have equal size
    var margin = {top: 10, right: 30, bottom: 20, left: 30};
    var width = 750 - margin.left - margin.right;
    var height = 350 - margin.top - margin.bottom;

    var marginFocus = {top: 10, right: 20, bottom: 100, left: 40};
    var marginContext = {top: 270, right: 20, bottom: 20, left: 40};

    var heightFocus = 350 - marginFocus.top - marginFocus.bottom;
    var heightContext = 350 - marginContext.top - marginContext.bottom;

    // we''ll define every single data source as global object, and we'll them as references for D3 charts
    var gScore = new Array();
    var gModel = new Array();
    var gGradient = new Array();
    var gMagnitude = new Array();

    // brush for scorechart zoom
    var brush;

    // we must ensure, that charts initialized only once
    var isInit = false;


    // each our chart is defined as array element
    var gSVG = new Array();
    var gXAxis = new Array();
    var gYAxis = new Array();

    var gX = new Array();
    var gY = new Array();
    var gXT = new Array();


    // elements of scorechart
    var focus;
    var context;
    var scoreData;
    var area2;

    var contains = function(needle) {
        // Per spec, the way to identify NaN is that it is not equal to itself
        var findNaN = needle !== needle;
        var indexOf;

        if(!findNaN && typeof Array.prototype.indexOf === 'function') {
            indexOf = Array.prototype.indexOf;
        } else {
            indexOf = function(needle) {
                var i = -1, index = -1;

                for(i = 0; i < this.length; i++) {
                    var item = this[i];

                    if((findNaN && item !== item) || item === needle) {
                        index = i;
                        break;
                    }
                }

                return index;
            };
        }

        return indexOf.call(this, needle) > -1;
    };

    function buildSurface() {
        /*
            TODO: to be implemented
        */
    }

    function appendHistogram(values,selector, id) {
        // A formatter for counts.
     //   if (id != "modelb") return;

      //  console.log("selector: " + selector + " id: " + id + " > " +values);

        var formatCount = d3.format(",.0f");
        var data = [];
        var binNum = 0;
        var binTicks = [];
        var min = null;
        var max = null;

        // convert json to d3 data structure
        var keys = Object.keys(values);
        for (var k = 0; k < keys.length; k++) {
            var key = keys[k];
            var fkey = parseFloat(key);
            var value = parseInt(values[key]);

            if (min == null) min = fkey;
            if (max == null) max = fkey;

            if (min > fkey) min = fkey;
            if (max < fkey) max = fkey;


            data.push({"x": parseFloat(key), "y": value});
            binTicks.push(key);
            binNum++;
        }

        var binWidth = parseFloat(width / (binNum - 1)) - 1;

        if (gSVG[id] != undefined || gSVG[id] != null) {
         //   console.log("SVG for key [" + id + "] is already defined. Going to update data");
/*
            var data = d3.layout.histogram()
                .bins(gX[id].ticks(20))
                (values);
*/



            gX[id] = d3.scale.linear()
                .domain([min, max])
                .range([0, width]);

            gXT[id] = d3.scale.linear()
                .domain([min, max])
                .range([0, width - margin.right - 5]);

            gY[id] = d3.scale.linear()
                            .domain([0, d3.max(data, function(d) { return d.y; })])
                            .range([height, 0]);

            gXAxis[id] = d3.svg.axis()
                            .scale(gX[id])
                            .orient("bottom")
                            .tickValues(binTicks);


            var bar = gSVG[id].selectAll(".bar")
                            .data(data)
                            .attr("transform", function(d) { return "translate(" + gXT[id](d.x) + "," + gY[id](d.y) + ")"; });

            gSVG[id].selectAll("text")
                .data(data)
                .attr("y", 6)
                .text(function(d) { return formatCount(d.y); });

            gSVG[id].selectAll("rect")
                .data(data)
                .attr("y", function(d) {
                    return 0;
                })
                .attr("height", function(d) { return height-gY[id](d.y) });

            gSVG[id].selectAll(".x.axis")
                            .attr("transform", "translate(0," + height + ")")
                            .call(gXAxis[id]);

            return;
        }


    //    console.log("SVG for key [" + id + "] is NOT defined");

        //var data = values;

/*
        if(isNaN(min)){
            min = 0.0;
            max = 1.0;
        }
*/



        gX[id] = d3.scale.linear()
                .domain([min, max])
                .range([0, width]);

        gXT[id] = d3.scale.linear()
                .domain([min, max])
                .range([0, width - margin.right - 5]);



        // Generate a histogram using twenty uniformly-spaced bins.
        /*var data = d3.layout.histogram()
                .bins(gX[id].ticks(20))
                (values);
        */

        /*
        console.log("---------------");
        console.log("Min: " + min + " Max: " +max );
        console.log("BinWidth: " + binWidth);
        console.log("BinTicks: " + binTicks);
        console.log("TicksNum: " + binTicks.length);
        console.log("Data: ");

        for (var i = 0; i < data.length; i++) {
            console.log("X: " + data[i].x + " Y: " + data[i].y);
        }
        console.log("---------------");
        */
        gY[id] = d3.scale.linear()
                .domain([0, d3.max(data, function(d) { return d.y; })])
                .range([height, 0]);



        gSVG[id] = d3.select(selector).append("svg")
                .attr("width", width + margin.left + margin.right)
                .attr("height", height + margin.top + margin.bottom)
                .append("g")
                .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        gXAxis[id] = d3.svg.axis()
                .scale(gX[id])
                .orient("bottom")
                .tickValues(binTicks);

        var bar = gSVG[id].selectAll(".bar")
                .data(data)
                .enter()
                .append("g")
                .attr("class", "bar")
                .attr("transform", function(d) { return "translate(" + gXT[id](d.x) + "," + gY[id](d.y) + ")"; });

        bar.append("rect")
                .attr("x", 1)
                .attr("y", 0)
                .attr("width", binWidth - 3)
                .attr("height", function(d) {
                        return height - gY[id](d.y);
                        });

        bar.append("text")
                .attr("dy", ".75em")
                .attr("y", 6)
                .attr("x", binWidth - (binWidth / 2))
                .attr("text-anchor", "middle")
                .attr("color","#000000")
                .attr("font-size","9px")
                .text(function(d) { return formatCount(d.y); });

        gSVG[id].append("g")
                .attr("class", "x axis")
                .attr("transform", "translate(0," + height + ")")
                .call(gXAxis[id]);
    }

 function brushed() {
   var valueline = d3.svg.line()
                        .x(function(d,i) { return gX["scorechart"](i); })
                        .y(function(d) { return gY["scorechart"](d); });
   gX["scorechart"].domain(brush.empty() ? gX["context"].domain() : brush.extent());
   focus.select(".line").attr("d", valueline(scoreData));
   focus.select(".x.axis").call(gXAxis["scorechart"]);
 }

 function appendLineChart(values,selector, id){
        scoreData = values;
        if (gSVG[id] != undefined || gSVG[id] != null) {
         //   console.log("SVG for scores [" + id + "] is already defined. Going to update data");

            var valueline = d3.svg.line()
                           .x(function(d,i) { return gX[id](i); })
                           .y(function(d) { return gY[id](d); });

            var max = d3.max(values);
            var min = d3.min(values);
            gX[id].domain([0,values.length]);
            gY[id].domain([min, max]);

            gX["context"].domain([0,values.length]);
            gY["context"].domain([min, max]);

            focus.select(".line")
                           .attr("d", valueline(values));

            focus.select(".x.axis")
                           .call(gXAxis[id]);

            focus.select(".y.axis")
                           .call(gYAxis[id]);

            area2 = d3.svg.area()
                        .interpolate("monotone")
                        .x(function(d,i) { return gX["context"](i); })
                        .y0(heightContext)
                        .y1(function(d) { return gY["context"](d); });

            context.select(".area")
                           .datum(values)
                           .attr("d", area2)


            context.select(".x.axis")
                            .attr("transform", "translate(0," + heightContext + ")")
                            .call(gXAxis["context"]);

            context.select(".y.axis")
                            .call(gYAxis["context"]);


            return;
        }

        // Set the ranges
        gX[id] = d3.scale.linear().range([0, width]);
        gY[id] = d3.scale.linear().range([heightFocus, 0]);

        gX["context"] = d3.scale.linear().range([0, width]);
        gY["context"] = d3.scale.linear().range([heightContext, 0]);

        // Adds the svg canvas
        gSVG[id] = d3.select(selector)
                .append("svg")
                .attr("width", width + margin.left + margin.right)
                .attr("height", height + margin.top + margin.bottom)
                .append("g")
              //  .attr("transform", "translate(" + marginFocus.left + "," + marginFocus.top + ")");


        focus = gSVG[id].append("g")
            .attr("class", "focus")
            .attr("transform", "translate(" + marginFocus.left + "," + marginFocus.top + ")");

        context = gSVG[id].append("g")
            .attr("class", "context")
            .attr("transform", "translate(" + marginContext.left + "," + marginContext.top + ")");


        brush = d3.svg.brush()
            .x(gX["context"])
            .on("brush", brushed);



        area2 = d3.svg.area()
            .interpolate("monotone")
            .x(function(d,i) { return gX["context"](i); })
            .y0(heightContext)
            .y1(function(d) { return gY["context"](d); });

        // Define the axes
        gXAxis[id]= d3.svg.axis().scale(gX[id])
//               .innerTickSize(-heightFocus)     //used as grid line
               .orient("bottom");//.ticks(5);

        gYAxis[id] = d3.svg.axis().scale(gY[id])
           //     .innerTickSize(-width)      //used as grid line
                .orient("left"); //.ticks(5);

        gYAxis["context"] = d3.svg.axis().scale(gY["context"]).orient("left").ticks(4);
        gXAxis["context"] = d3.svg.axis().scale(gX["context"]).orient("bottom");


        // Define the line
        var valueline = d3.svg.line()
                .x(function(d,i) { return gX[id](i); })
                .y(function(d) { return gY[id](d); });

        // Scale the range of the data
        var max = d3.max(values);
        var min = d3.min(values);
        gX[id].domain([0,values.length]);
        gY[id].domain([min, max]);

        gX["context"].domain([0,values.length]);
        gY["context"].domain([min, max]);

        // Add the valueline path.
        focus.append("path")
                .attr("class", "line")
                .attr("d",  valueline(values));


        // Add the X Axis
        focus.append("g")
                .attr("class", "x axis")
                .attr("transform", "translate(0," + heightFocus + ")")
                .call(gXAxis[id]);

        // Add the Y Axis
        focus.append("g")
                .attr("class", "y axis")
                .call(gYAxis[id]);


        context.append("path")
                      .datum(values)
                      .attr("class", "area")
                      .attr("d", area2);

        context.append("g")
                .attr("class", "x axis")
                .attr("transform", "translate(0," + heightContext + ")")
                .call(gXAxis["context"]);

        // Add the Y Axis
        context.append("g")
                .attr("class", "y axis")
                .call(gYAxis["context"]);

        context.append("g")
              .attr("class", "x brush")
              .call(brush)
            .selectAll("rect")
              .attr("y", -6)
              .attr("height", heightContext + 7);
    }

    function appendMultiLineChart(map,selector, id){
        if (gSVG[id] != undefined || gSVG[id] != null) {
                var valueline = d3.svg.line()
                                .x(function(d,i) { return gX[id](i); })
                                .y(function(d) { return gY[id](d); });


                var max = -Number.MAX_VALUE;
                var size = 1;
                for( var key in map ){
                    var values = map[key];
                    var thisMax = d3.max(values);
                    if(thisMax > max) max = thisMax;
                    size = values.length;
                }
                gX[id].domain([0,size]);
                gY[id].domain([0, max]);

                var color = d3.scale.category10();
                var i=0;
                for(var key in map ){
                    var values = map[key];
                    gSVG[id].select(".line.l"+i)
                            .attr("d", valueline(values));
                            i++;
                }

                gSVG[id].select(".x.axis")
                        .call(gXAxis[id]);

                gSVG[id].select(".y.axis")
                        .call(gYAxis[id]);

                return;
        }

        var keys = Object.keys(map)
        if(keys.length == 0) return;    //nothing to plot


        // Set the ranges
        gX[id] = d3.scale.linear().range([0, width]);
        gY[id] = d3.scale.linear().range([height, 0]);

        // Define the axes
        gXAxis[id] = d3.svg.axis().scale(gX[id])
                .innerTickSize(-height)     //used as grid line
                .orient("bottom").ticks(5);

        gYAxis[id] = d3.svg.axis().scale(gY[id])
                .innerTickSize(-width)      //used as grid line
                .orient("left").ticks(5);

        // Define the line
        var valueline = d3.svg.line()
                .x(function(d,i) { return gX[id](i); })
                .y(function(d) { return gY[id](d); });

        // Adds the svg canvas
        gSVG[id] = d3.select(selector)
                .append("svg")
                .attr("width", width + margin.left + margin.right)
                .attr("height", height + margin.top + margin.bottom)
                .append("g")
                .attr("transform",
                "translate(" + margin.left + "," + margin.top + ")");

        // Scale the range of the data
        var max = -Number.MAX_VALUE;
        var size = 1;
        for( var key in map ){
            var values = map[key];
            var thisMax = d3.max(values);
            if(thisMax > max) max = thisMax;
            size = values.length;
        }
        gX[id].domain([0,size]);
        gY[id].domain([0, max]);

        // Add the valueline path.
        var color = d3.scale.category10();
        var i=0;
        for( var key in map ){
            var values = map[key];
            gSVG[id].append("path")
                .attr("class", "line l" + i)
                .style("stroke", color(i))
                .attr("d", valueline(values));
            i++;
        }

        // Add the X Axis
        gSVG[id].append("g")
                .attr("class", "x axis")
                .attr("transform", "translate(0," + height + ")")
                .call(gXAxis[id]);

        // Add the Y Axis
        gSVG[id].append("g")
                .attr("class", "y axis")
                .call(gYAxis[id]);

        //Add legend
        var legendSpace = width/i;
        i = 0;
        for( var key in map ){
            var values = map[key];
            var last = values[values.length-1];
            var toDisplay = key + " (" + last.toPrecision(5) + ") ";
            gSVG[id].append("text")
                .attr("x", (legendSpace/2)+i*legendSpace) // spacing
                .attr("y", height + (margin.bottom/2)+ 5)
                .attr("class", "legend")    // style the legend
                .style("fill", color(i))
                .text(toDisplay);

            i++;
        }
    }

var timed = function() {

                    var sid = getParameterByName("sid");
                    if (sid == undefined) sid = 0;

                    $.ajax({
                        url:"/weights" + "/updated?sid=" + sid,
                        async: true,
                        error: function (query, status, error) {
                            $.notify({
                                title: '<strong>No connection!</strong>',
                                message: 'DeepLearning4j UiServer seems to be down!'
                            },{
                                type: 'danger',
                                placement: {
                                    from: "top",
                                    align: "center"
                                    },
                            });
                            setTimeout(timed, 10000);
                        },
                        success: function( data ) {
                                    /*
                                        /weights/data should be changed to /weights/data/{time} and only delta should be passed over network
                                    */
                                    d3.json("/weights"+'/data?sid=' + sid,function(error,json) {

                                        //Get last update time; do nothing if not a new update
                                        var updateTime = json['lastUpdateTime'];
                                        var lastUpdateTime = $('#lastupdate .updatetime').text();
                                        if(updateTime == lastUpdateTime) return;


                                        var model = json['parameters'];
                                        var gradient = json['gradients'];
                                        var score = json['score'];
                                        var scores = json['scores'];
                                        var updateMagnitudes = json['updateMagnitudes'];
                                        var paramMagnitudes = json['paramMagnitudes'];


                                        // !gradient was removed here, because for spark models it can be absen
                                        // !updateMagnitudes
                                        if(!model  || !score || !scores || !paramMagnitudes ) {
                                            console.log("Model: " + model);
                                            console.log("Score: " + score);
                                            console.log("Scores: " + scores);
                                            console.log("ParamMagnitudes: " + paramMagnitudes);
                                            setTimeout(timed, 10000);
                                            $.notify({
                                                    	title: '<strong>No data available!</strong>',
                                                    	message: 'Please check, if <strong>HistogramTrainingListener</strong> was enabled.'
                                                    },{
                                                    	type: 'danger',
                                                    	placement: {
                                                        		from: "top",
                                                        		align: "center"
                                                        	},
                                                    });
                                            return;
                                        }
                                        $('#score').html('' + parseFloat(score).toFixed(3));

                                        /*
                                        <div style="">
                                            Current score: <b><span class="score" id="score"></span></b><br />
                                        </div>
                                        */
                                        //$('#scores .chart').html('');
                                        if (gSVG["scorechart"] == undefined || gSVG["scorechart"] == null) {
                                            $("#schart").html('<div class="scoreboard">Current score:<br/><b><span class="score" id="score">0.0</span></b></div>');
                                            var scdiv = '<div id="scorechart" class="scorechart"></div>';
                                            $('#scores .chart').append(scdiv);
                                        }
                                        appendLineChart(scores,'#scores .chart', "scorechart");

                                        //clear out body of where the chart content will go
                                        //$('#model .charts').html('');
                                        //$('#gradient .charts').html('');
                                        var keys = Object.keys(model);
                                        for(var i = 0; i < keys.length; i++) {
                                            var key = keys[i];
                                            //model id class charts
                                            var selectorModel = '#model .charts';
                                            var selectorGradient = '#gradient .charts';
                                            //append div to each node where the chart content will
                                            //go and pass that in to the chart renderer

                                            // we append divs only once
                                            if (gSVG["model"+ key] == undefined || gSVG["model"+ key ] == null) {
                                                var divModel = '<div id="model'+ key+'" class="' + key + '" style="' +  ((visibleModel == key) ? "visibility: visible; display: block;" : "visibility: hidden; display: none;") +';"></div>';
                                                var divGradient = '<div id="gradient'+ key+'" class="' + key + '" style="' +  ((visibleGradient == key) ? "visibility: visible; display: block;" : "visibility: hidden; display: none;") +';"></div>';
                                                $(selectorModel).append(divModel);
                                                if (gradient != undefined  && gradient[key] != undefined) $(selectorGradient).append(divGradient);
                                            }

                                            if (model[key] != undefined) appendHistogram(model[key],selectorModel + ' .' + key, "model"+ key );
                                            if (gradient != undefined && gradient[key] != undefined) appendHistogram(gradient[key],selectorGradient + ' .' + key, "gradient"+ key );
                                            /*
                                                update selector box if needed
                                            */
                                            if (!contains.call(modelSelector, key)) {
                                      //          console.log("Adding model selector: " + key);
                                                modelSelector.push(key);

                                                $("#modelSelector").append("<option value='"+ key+"'>" + key + "</option>");
                                                $("#gradientSelector").append("<option value='"+ key+"'>" + key + "</option>");
                                            }
                                        }


                                        //Plot mean magnitudes: weights and params
                                        //$('#magnitudes .charts').html('');


                                        if(paramMagnitudes  != undefined) {
                                            console.log("ParamMag length: " + paramMagnitudes.length);
                                            for(var i=0; i<paramMagnitudes.length; i++ ){
                                                //Maps:
                                                var mapParams = i < paramMagnitudes.length ? paramMagnitudes[i] : 0;
                                                var mapUpdates = i < updateMagnitudes.length ? updateMagnitudes[i] : 0;

                                                var selectorModel = '#magnitudes .charts'

                                                // we create divs only once
                                                if (gSVG["layer" + i + "param"] == undefined || gSVG["layer" + i + "param"] == null) {
                                                    var div = '<div id="layer' + i + 'param" class="layer' + i + 'param" style="' +  ((visibleMagnitude == "layer" + i + "param" ) ? "visibility: visible; display: block;" : "visibility: hidden; display: none;") +';"></div>';
                                                    if (i < paramMagnitudes.length) $(selectorModel).append(div);
                                                    div = '<div id="layer' + i + 'grad" class="layer' + i + 'grad" style="' +  ((visibleMagnitude == "layer" + i + "grad" ) ? "visibility: visible; display: block;" : "visibility: hidden; display: none;") +';"></div>';
                                                    if (i < updateMagnitudes.length) $(selectorModel).append(div);
                                                }

                                                var key = "layer" + i + "param";
                                                if (i < paramMagnitudes.length) appendMultiLineChart(mapParams,selectorModel + ' .layer' + i + 'param',"layer" + i + "param");
                                                if (i < updateMagnitudes.length) appendMultiLineChart(mapUpdates,selectorModel + ' .layer' + i + 'grad',"layer" + i + "grad");

                                                if (!contains.call(magnitudesSelector, key)) {
                                                    //  console.log("Adding magnitudes selector: " + key);
                                                    magnitudesSelector.push(key);

                                                    if (i < paramMagnitudes.length) $("#magnitudeSelector").append("<option value='layer" + i + "param'>Layer " + i + " Parameter Mean Magnitudes</option>");
                                                    if (i < updateMagnitudes.length) $("#magnitudeSelector").append("<option value='layer" + i + "grad'>Layer " + i + " Update/Gradient Mean Magnitudes</option>");
                                                }
                                            }
                                        }

                                        // this hack allows first selection of visible model histo
                                        if (visibleModel == "") {
                                            $("#modelSelector").val($("#modelSelector option:first").val());
                                            selectModel();
                                        }

                                        if (visibleGradient == "") {
                                            $("#gradientSelector").val($("#gradientSelector option:first").val());
                                            selectGradient();
                                        }

                                        if (visibleMagnitude == "") {
                                            $("#magnitudeSelector").val($("#magnitudeSelector option:first").val());
                                            selectMagnitude();
                                        }

                                        var time = new Date(updateTime);
                                        $('#updatetime').html(time.customFormat("#DD#/#MM#/#YYYY# #hhh#:#mm#:#ss#"));

                                        var grad = Object.keys(gradient);

                                         if (gradient == undefined || grad.length == 0) {
                                                // if gradient  isn't available, it just means we're in spark mode, without gradients
                                                console.log("Calling noGrad");
                                                var nograd = "<div style='position: absolute; text-align: center; top: 50%; left: 50%; width: 500px;  -webkit-transform: translate(-50%, -50%); transform: translate(-50%, -50%);  '><strong>Gradients are unavailable in Spark mode</strong></div>";
                                                $("#gradient").html(nograd);
                                          }

                                        // all subsequent refreshes are delayed by 2 seconds
                                        // TODO: make this configurable
                                        setTimeout(timed, 2000)
                                    });
                    }
                })
            };

// first update is fired almost immediately, 2s timeout
setTimeout(timed,2000);


    function selectModel() {
      //  console.log("Switching off model view: " + visibleModel);
        if (visibleModel != "") {
            $("#model" + visibleModel).css("visibility","hidden");
            $("#model" + visibleModel).css("display","none");
        }

        visibleModel = $("#modelSelector").val();
        $("#model" + visibleModel).css("visibility","visible");
        $("#model" + visibleModel).css("display","block");
     //   console.log("Switching on model view:" + visibleModel);
    }

    function selectGradient() {
       //     console.log("Switching off gradient view: " + visibleGradient);
            if (visibleGradient != "") {
                $("#gradient" + visibleGradient).css("visibility","hidden");
                $("#gradient" + visibleGradient).css("display","none");
            }

            visibleGradient = $("#gradientSelector").val();
            $("#gradient" + visibleGradient).css("visibility","visible");
            $("#gradient" + visibleGradient).css("display","block");
      //      console.log("Switching on gradient view:" + visibleGradient);
    }

    function selectMagnitude() {
    //    console.log("Switching off magnitude view: " + visibleMagnitude);
        if (visibleMagnitude != "") {
            $("#" + visibleMagnitude).css("visibility","hidden");
            $("#" + visibleMagnitude).css("display","none");
        }

        visibleMagnitude = $("#magnitudeSelector").val();

        $("#" + visibleMagnitude).css("visibility","visible");
        $("#" + visibleMagnitude).css("display","block");
    //    console.log("Switching on magnitude view:" + visibleMagnitude);
    }

