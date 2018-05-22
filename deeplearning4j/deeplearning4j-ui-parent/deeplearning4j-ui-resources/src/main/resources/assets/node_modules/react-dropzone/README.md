react-dropzone
=================

Simple HTML5 drag-drop zone in React.js.

[![Screenshot of Dropzone](https://raw.githubusercontent.com/paramaggarwal/react-dropzone/master/screenshot.png)](http://paramaggarwal.github.io/react-dropzone/)

Demo: http://paramaggarwal.github.io/react-dropzone/
Usage
=====

Simply `require('react-dropzone')` and specify an `onDrop` method that accepts an array of dropped files. You can customize the content of the Dropzone by specifying children to the component.

You can also specify a `style` object to apply to the `Dropzone` component. Optionally pass in a `size` property to configure the size of the Dropzone.

```jsx

/** @jsx React.DOM */
var React = require('react');
var Dropzone = require('react-dropzone');

var DropzoneDemo = React.createClass({
    onDrop: function (files) {
      console.log('Received files: ', files);
    },

    render: function () {
      return (
          <div>
            <Dropzone onDrop={this.onDrop} size={150} >
              <div>Try dropping some files here, or click to select files to upload.</div>
            </Dropzone>
          </div>
      );
    }
});

React.render(<DropzoneDemo />, document.body);
```

Author
======

Param Aggarwal (paramaggarwal@gmail.com)

License
=======

MIT
