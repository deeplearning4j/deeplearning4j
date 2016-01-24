<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <title>Neural Network activations</title>
    <link href='http://fonts.googleapis.com/css?family=Roboto:400,300' rel='stylesheet' type='text/css'>

    <!-- Latest compiled and minified CSS -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css" integrity="sha384-1q8mTJOASx8j1Au+a5WDVnPi2lkFfwwEAa8hDDdjZlpLegxhjVME1fgjWPGmkzs7" crossorigin="anonymous" />

    <!-- Optional theme -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap-theme.min.css" integrity="sha384-fLW2N01lMqjakBkx3l/M9EahuwpSfeNvV63J5ezn3uZzapT0u7EYsXMjQV+0En5r" crossorigin="anonymous" />

    <!-- Latest compiled and minified JavaScript -->
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/js/bootstrap.min.js" integrity="sha384-0mSbJDEHialfmuBBQP6A4Qrprq5OVfW37PRR3j5ELqxss1yVqOtnepnHVP9aJ7xS" crossorigin="anonymous"></script>

    <!-- jQuery -->
    <script src="https://code.jquery.com/jquery-2.2.0.min.js"></script>
    <style>
        body {
        font-family: 'Roboto', sans-serif;
        color: #333;
        font-weight: 300;
        font-size: 16px;
        }
        .hd {
        background-color: #000000;
        font-size: 18px;
        color: #FFFFFF;
        }
        .block {
        width: 250px;
        height: 350px;
        display: inline-block;
        border: 1px solid #DEDEDE;
        margin-right: 64px;
        }
        .hd-small {
        background-color: #000000;
        font-size: 14px;
        color: #FFFFFF;
        }
    </style>
    <script type="text/javascript">
        setInterval(function() {
            var d = new Date();
            $("#pic").attr("src", "/activations/img?"+d.getTime());
        },3000);
    </script>

</head>



<body>
<table style="width: 100%; padding: 5px;" class="hd">
    <tbody>
    <tr>
        <td style="width: 48px;"><img src="/assets/deeplearning4j.img"  border="0"/></td>
        <td>DeepLearning4j UI</td>
        <td style="width: 128px;">&nbsp; <!-- placeholder for future use --></td>
    </tr>
    </tbody>
</table>
<br /><br />
<div style="width: 100%; text-align: center">
    <div id="embed" style="display: inline-block;">
        <img src="/activations/img" id="pic"/>
    </div>
</div>
</body>

</html>