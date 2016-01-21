<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Activation renders 3</title>
    <script src="//ajax.googleapis.com/ajax/libs/jquery/1/jquery.min.js"></script>
    <script type="text/javascript">
        setInterval(function() {
            var d = new Date();
            $("#pic").attr("src", "/activations/img?"+d.getTime());
        },3000);
    </script>
    <style type="text/css">
    </style>
</head>



<body>
<div id="embed">
    <img src="/activations/img" id="pic"/>
</div>

</body>

</html>