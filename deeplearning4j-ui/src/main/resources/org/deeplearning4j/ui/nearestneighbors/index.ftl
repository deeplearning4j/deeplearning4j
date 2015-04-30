<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Nearest Neighbors</title>
    <link rel="stylesheet" href="assets/bootstrap-3.3.4-dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="assets/bootstrap-3.3.4-dist/css/bootstrap-theme.min.css">
    <link rel="stylesheet" href="assets/css/simple-sidebar.css">
    <link rel="stylesheet" href="assets/css/style.css">
    <script src="https://code.jquery.com/jquery-2.1.3.min.js"></script>
    <script src="assets/jquery-fileupload.js"></script>
    <script src="assets/bootstrap-3.3.4-dist/js/bootstrap.min.js"></script>
    <script src="assets/js/nearestneighbors/app.js"></script>
</head>

<body>
<div id="container">
    <div id="wrapper">

        <div id="sidebar-wrapper"></div>
        <div id="page-content-wrapper">
            <div class="container-fluid" style="text-align: center">
                <h1>Deeplearning4j</h1>
                <h2>k Nearest Neighbors</h2>
                <h4>Upload and submit a <b><i>vectorized</i></b> text file. <br>
                    Then select a word on the left panel. <br>
                    A list of nearest neighbors will appear on this page.</h4>
                <br>
                <div class="row" id="upload" style="text-align:center; margin: auto">
                    <form encType="multipart/form-data" action="/nearestneighbors/upload" method="POST" id="form" class="btn btn-default">
                        <input name="file" type="file" class="btn-file">
                        <br>
                        <input type="submit">
                    </form>
                </div>
                <div id="neighbors">
                </div>
            </div>
        </div>

    </div>
</div>
</body>
</html>
