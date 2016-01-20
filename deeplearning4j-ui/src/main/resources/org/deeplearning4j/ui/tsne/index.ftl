<!DOCTYPE html>
<html lang="en" data-framework="react">
	<head>
		<meta charset="utf-8">
		<title>TSNE</title>
		<link rel="stylesheet" href="node_modules/todomvc-common/base.css">
		<link rel="stylesheet" href="node_modules/todomvc-app-css/index.css">

        <!-- jQuery -->
        <script src="//ajax.googleapis.com/ajax/libs/jquery/1/jquery.min.js"></script>
        <script src="/assets/d3.min.js"></script>
        <script src="/assets/render.js"></script>
        <link href='http://fonts.googleapis.com/css?family=Roboto:400,300' rel='stylesheet' type='text/css'>




        <style>
            body {
                font-family: 'Roboto', sans-serif;
                color: #333;
                font-weight: 300;
                font-size: 16px;
            }
            svg {
                border: 1px solid #333;
            }
            #wrap {
                width: 800px;
                margin-left: auto;
                margin-right: auto;
            }
            #embed {
                margin-top: 10px;
            }
            h1 {
                text-align: center;
                font-weight: normal;
            }
            .tt {
                margin-top: 10px;
                background-color: #EEE;
                border-bottom: 1px solid #333;
                padding: 5px;
            }
            .txth {
                color: #F55;
            }
            .cit {
                font-family: courier;
                padding-left: 20px;
                font-size: 14px;
            }
        </style>

        <script>


        </script>

	</head>
	<body>
		<section id="todoapp"></section>
		<footer id="info">
			<p>Double-click to edit a todo</p>
			<p>Created by <a href="http://github.com/petehunt/">petehunt</a></p>
			<p>Part of <a href="http://todomvc.com">TodoMVC</a></p>
		</footer>

		<script src="node_modules/todomvc-common/base.js"></script>
		<script src="node_modules/react/dist/react-with-addons.js"></script>
		<script src="node_modules/react/dist/JSXTransformer.js"></script>
		<script src="node_modules/director/build/director.js"></script>

		<script src="js/utils.js"></script>
		<script src="js/todoModel.js"></script>
		<!-- jsx is an optional syntactic sugar that transforms methods in React's
		`render` into an HTML-looking format. Since the two models above are
		unrelated to React, we didn't need those transforms. -->
		<script type="text/jsx" src="js/todoItem.jsx"></script>
		<script type="text/jsx" src="js/footer.jsx"></script>
		<script type="text/jsx" src="js/app.jsx"></script>
	</body>
</html>
