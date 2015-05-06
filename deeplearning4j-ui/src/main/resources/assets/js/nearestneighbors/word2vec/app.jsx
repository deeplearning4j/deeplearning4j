/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

/*jshint quotmark:false */
/*jshint white:false */
/*jshint trailing:false */
/*jshint newcap:false */
/*global React, Router*/

(function () {
	'use strict';

	/** @jsx React.DOM */

	var FileForm = React.createClass({
		getInitialState: function() {
			return {data_uri: null}
		},
		handleSubmit: function() {
			$.ajax({
				url: '/nearestneighbors/upload',
				type: "POST",
				contentType : 'multipart/form-data',
				data: this.state.data_uri,
				success: function(data) {
					// do stuff
				}.bind(this),
				error: function(xhr, status, err) {
					// do stuff
				}.bind(this)
			});
			return false;
		},
		handleFile: function(e) {
			var reader = new FileReader();
			var file = e.target.files[0];

			reader.onload = function(upload) {
				this.setState({
					data_uri: upload.target.result
				});
				console.log(this.state.data_uri);
			}.bind(this);

			reader.readAsDataURL(file);
		},
		render: function() {
			return (
				<form onSubmit={this.handleSubmit} encType="multipart/form-data">
					<input type="file" onChange={this.handleFile} />
					<input type="submit" value="Submit" />
            </form>

        );
    }
});


	React.render(<FileForm />, document.getElementById('upload'));


})();
