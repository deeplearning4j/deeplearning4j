<!doctype html>
<html lang="en" data-framework="react">
	<head>
		<meta charset="utf-8">
		<title>Renders</title>
		<link rel="stylesheet" href="assets/node_modules/todomvc-common/base.css">
		<link rel="stylesheet" href="assets/node_modules/todomvc-app-css/index.css">
	</head>
	<body>
		<section id="todoapp"></section>
		<footer id="info">
			<p>Double-click to edit a todo</p>
			<p>Created by <a href="http://github.com/petehunt/">petehunt</a></p>
			<p>Part of <a href="http://todomvc.com">TodoMVC</a></p>
		</footer>

		<script src="assets/node_modules/todomvc-common/base.js"></script>
		<script src="assets/node_modules/react/dist/react-with-addons.js"></script>
		<script src="assets/node_modules/react/dist/JSXTransformer.js"></script>
		<script src="assets/node_modules/director/build/director.js"></script>

		<script src="assets/js/utils.js"></script>
		<script src="assets/js/todoModel.js"></script>
		<!-- jsx is an optional syntactic sugar that transforms methods in React's
		`render` into an HTML-looking format. Since the two models above are
		unrelated to React, we didn't need those transforms. -->
		<script type="text/jsx" src="assets/js/todoItem.jsx"></script>
		<script type="text/jsx" src="assets/js/footer.jsx"></script>
		<script type="text/jsx" src="assets/js/app.jsx"></script>
	</body>
</html>
