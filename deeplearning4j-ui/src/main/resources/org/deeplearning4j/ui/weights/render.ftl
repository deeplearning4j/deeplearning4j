<!DOCTYPE html>
<meta charset="utf-8">
<style>

    body {
        font: 10px sans-serif;
    }

    .bar rect {
        fill: steelblue;
        shape-rendering: crispEdges;
    }

    .bar text {
        fill: #fff;
    }

    .axis path, .axis line {
        fill: none;
        stroke: #000;
        shape-rendering: crispEdges;
    }

</style>
<body>
<script src="//ajax.googleapis.com/ajax/libs/jquery/1/jquery.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js"></script>
<script>
    function appendHistogram(values,selector) {
        // A formatter for counts.
        var formatCount = d3.format(",.0f");

        var margin = {top: 10, right: 30, bottom: 30, left: 30},
                width = 960 - margin.left - margin.right,
                height = 500 - margin.top - margin.bottom;
        var data = values;
        var min = d3.min(data);
        var max = d3.max(data);
        if(isNaN(min)){
            min = 0.0;
            max = 1.0;
        }

        var x = d3.scale.linear()
                .domain([min, max])
                .range([0, width]);

        // Generate a histogram using twenty uniformly-spaced bins.
        var data = d3.layout.histogram()
                .bins(x.ticks(20))
                (values);

        var y = d3.scale.linear()
                .domain([0, d3.max(data, function(d) { return d.y; })])
                .range([height, 0]);

        var xAxis = d3.svg.axis()
                .scale(x)
                .orient("bottom");

        var svg = d3.select(selector).append("svg")
                .attr("width", width + margin.left + margin.right)
                .attr("height", height + margin.top + margin.bottom)
                .append("g")
                .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        var bar = svg.selectAll(".bar")
                .data(data)
                .enter().append("g")
                .attr("class", "bar")
                .attr("transform", function(d) { return "translate(" + x(d.x) + "," + y(d.y) + ")"; });
        
        bar.append("rect")
                .attr("x", 1)
                .attr("width", x(min+data[0].dx) -1 )
                .attr("height", function(d) { return height - y(d.y); });

        bar.append("text")
                .attr("dy", ".75em")
                .attr("y", 6)
                .attr("x", x(min+data[0].dx) / 2)
                .attr("text-anchor", "middle")
                .text(function(d) { return formatCount(d.y); });

        svg.append("g")
                .attr("class", "x axis")
                .attr("transform", "translate(0," + height + ")")
                .call(xAxis);
    }


    setInterval(function() {

        $.get( "/weights/updated", function( data ) {
                d3.json('/weights/data',function(error,json) {
                    var model = json['parameters'];
                    var gradient = json['gradients'];
                    var score = json['score'];
                    if(!model || !gradient || !score)
                        return;
                    $('.score').html('' + score);
                    //clear out body of where the chart content will go
                    $('#model .charts').html('');
                    $('#gradient .charts').html('');
                    var keys = Object.keys(model);
                    for(var i = 0; i < keys.length; i++) {
                        var key = keys[i];
                        //model id class charts
                        var selectorModel = '#model .charts';
                        var selectorGradient = '#gradient .charts';
                        //append div to each node where the chart content will
                        //go and pass that in to the chart renderer
                        var div = '<div class="' + key + '"><h4>' + key + '</h4></div>';
                        $(selectorModel).append(div);
                        $(selectorGradient).append(div);
                        appendHistogram(model[key]['dataBuffer'],selectorModel + ' .' + key);
                        appendHistogram(gradient[key]['dataBuffer'],selectorGradient + ' .' + key);
                    }
                });

        });

    },1000);



</script>

<body>
<div id="score">
    <h4>Score</h4>
    <div class="score"></div>
</div>
<div id="model">
    <h4>Model</h4>
    <div class="charts"></div>
</div>
<div id="gradient">
    <h4>Gradient</h4>
    <div class="charts"></div>
</div>
</body>