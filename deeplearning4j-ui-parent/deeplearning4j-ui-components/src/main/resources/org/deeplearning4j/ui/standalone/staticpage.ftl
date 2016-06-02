<html>
<head>
    <style type="text/css">
        html, body {
            width: 100%;
            height: 100%;
        }

        .bgcolor {
            background-color: #FFFFFF;
        }

        .hd {
            background-color: #000000;
            font-size: 18px;
            color: #FFFFFF;
        }

        .sectionheader {
            background-color: #888888;
            width:100%;
            font-size: 16px;
            font-style: bold;
            color: #FFFFFF;
            /*padding-left: 40px;*/
            /*padding-right: 8px;*/
            /*padding-top: 2px;*/
            /*padding-bottom: 2px;*/

        }

        .subsectiontop {
            background-color: #F5F5FF;
            height: 300px;
        }

        .subsectionbottom {
            background-color: #F5F5FF;
            height: 540px;
        }

        h1 {
            font-family: Georgia, Times, 'Times New Roman', serif;
            font-size: 28px;
            font-style: bold;
            font-variant: normal;
            font-weight: 500;
            line-height: 26.4px;
        }

        h3 {
            font-family: Georgia, Times, 'Times New Roman', serif;
            font-size: 16px;
            font-style: normal;
            font-variant: normal;
            font-weight: 500;
            line-height: 26.4px;
        }

        div.outerelements {
            padding-bottom: 30px;
        }

        /** Line charts */
        path {
            stroke: steelblue;
            stroke-width: 2;
            fill: none;
        }

        .axis path, .axis line {
            fill: none;
            stroke: #000;
            shape-rendering: crispEdges;
        }

        .tick line {
            opacity: 0.2;
            shape-rendering: crispEdges;
        }

        /** Bar charts */
        .bar {
            fill: steelblue;
        }

        rect {
            fill: steelblue;
        }

        .legend rect {
            fill:white;
            stroke:black;
            opacity:0.8;
        }

    </style>
    <title>Data Analysis</title>

</head>
<body style="padding: 0px; margin: 0px" onload="generateContent()">

<#--<meta name="viewport" content="width=device-width, initial-scale=1">-->
<link href="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap.min.css">
<script src="http://ajax.googleapis.com/ajax/libs/jquery/1/jquery.min.js"></script>
<link href="http://code.jquery.com/ui/1.11.4/themes/smoothness/jquery-ui.css">
<script src="http://code.jquery.com/jquery-1.10.2.js"></script>
<script src="http://code.jquery.com/ui/1.11.4/jquery-ui.js"></script>
<script src="http://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js"></script>
<script src="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/js/bootstrap.min.js"></script>
<script>

    function generateContent(){
        var mainDiv = $('#outerdiv');

        <#list components as c>
            var div_${c.id} = $('#${c.id}');
            var html_${c.id} = div_${c.id}.html();

            var component = Component.getComponent(html_${c.id});
            component.render(mainDiv);
        </#list>

    }
</script>

<script>
    ${scriptcontent}
</script>

<div style="width:1400px; margin:0 auto; border:0px" id="outerdiv">

</div>

<#list components as c>
<div id="${c.id}" style="display:none">
${c.content}
</div>
</#list>

</body>

</html>