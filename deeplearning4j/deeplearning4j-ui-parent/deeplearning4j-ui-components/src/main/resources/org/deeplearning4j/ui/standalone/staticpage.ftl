<html>
<head>
    <style type="text/css">
        html, body {
            width: 100%;
            height: 100%;
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