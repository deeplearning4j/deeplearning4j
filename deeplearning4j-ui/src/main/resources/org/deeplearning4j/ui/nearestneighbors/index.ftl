<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Nearest Neighbors</title>
    <link rel="stylesheet" href="/assets/bootstrap-3.3.4-dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="/assets/bootstrap-3.3.4-dist/css/bootstrap-theme.min.css">
    <link rel="stylesheet" href="/assets/css/simple-sidebar.css">
    <link rel="stylesheet" href="/assets/css/style.css">
    <script src="https://code.jquery.com/jquery-2.1.3.min.js"></script>
    <script src="/assets/jquery-fileupload.js"></script>
    <script src="/assets/bootstrap-3.3.4-dist/js/bootstrap.min.js"></script>
    <script src="/assets/js/nearestneighbors/app.js"></script>
</head>

<body>
<div id="container">
    <div id="wrapper">

        <div id="sidebar-wrapper"></div>
        <div id="page-content-wrapper">
            <div class="container-fluid">
                <h1 style="text-align: center; font-size: 400%">Deeplearning4j</h1>
                <hr>
                <h2>k Nearest Neighbors</h2>
                <ul>
                    <li>Upload a <b><i>vectorized</i></b> text file. OR enter a url to get data from.</li>
                    <ul>
                        <li>The text file should be space-delimited.</li>
                        <li>Each row should be a feature vector separated by spaces.</li>
                        <li>If an individual feature has multiple words, use underscore to separate the words.</li>
                    </ul>
                    <li>Enter an integer value for k (number of nearest neighbors).</li>
                    <li>Then select a word on the left panel.</li>
                    <li>A list of k nearest neighbors will appear on this page.</li>
                    <li>Optional: Select a new word to update nearest neighbors.</li></ul>
                <br>

                <div class="row" id="upload">
                    <form encType="multipart/form-data" action="/nearestneighbors/upload" method="POST" id="form">
                        <input name="file" type="file">
                        <br>
                        <input type="submit">
                    </form>
                </div>
                <div class="row" id="url">
                    <label for="url">Enter a url</label>
                    <input type="text" id="urlval">
                    <button value="Submit" id="urlsubmit">Submit</button>
                </div>

                <div class="row" id="kform">
                    Enter an integer value for k: <input type="text" name="k" id="k" value="5">
                </div>
                <div id="neighbors">
                </div>
            </div>
        </div>

    </div>
</div>
</body>
</html>
