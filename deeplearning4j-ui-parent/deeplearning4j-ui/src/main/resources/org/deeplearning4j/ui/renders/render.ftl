<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Filter renders 2</title>
    <script src="//ajax.googleapis.com/ajax/libs/jquery/1/jquery.min.js"></script>
    <script type="text/javascript">
        setInterval(function() {
            var d = new Date();
            $("#pic").attr("src", "/filters/img?"+d.getTime());
        },1000);
    </script>
</head>

<body>
<div id="embed">
    <img src="/filters/img" id="pic"/>
</div>

</body>

</html>