/**
 * fileUpload
 * http://abandon.ie
 *
 * Plugin to add file uploads to jQuery ajax form submit
 *
 * November 2013
 *
 * @version 0.9
 * @author Abban Dunne http://abandon.ie
 * @license MIT
 *
 */
;(function($, window, document, undefined)
{
    // Create the defaults once
    var pluginName = "fileUpload",
        defaults = {
            uploadData    : {},
            submitData    : {},
            uploadOptions : {},
            submitOptions : {},
            before        : function(){},
            beforeSubmit  : function(){ return true; },
            success       : function(){},
            error         : function(){},
            complete      : function(){}
        };

    // The actual plugin constructor
    function Plugin(element, options)
    {
        this.element    = element;
        this.$form      = $(element);
        this.$uploaders = $('input[type=file]', this.element);
        this.files      = {};
        this.settings   = $.extend({}, defaults, options);
        this._defaults  = defaults;
        this._name      = pluginName;
        this.init();
    }

    Plugin.prototype =
    {
        /**
         * Initialize the plugin
         *
         * @return void
         */
        init: function()
        {
            this.$uploaders.on('change', { context : this }, this.processFiles);
            this.$form.on('submit', { context : this }, this.uploadFiles);
        },



        /**
         * Process files after they are added
         *
         * @param  jQuery event
         * @return void
         */
        processFiles: function(event)
        {
            var self = event.data.context;
            self.files[$(event.target).attr('name')] = event.target.files;
        },



        /**
         * Handles the file uploads
         *
         * @param  jQuery event
         * @return void
         */
        uploadFiles: function(event)
        {
            event.stopPropagation(); // Stop stuff happening
            event.preventDefault(); // Totally stop stuff happening

            var self = event.data.context;

            // Run the before callback
            self.settings.before();

            // Declare a form data object
            var data = new FormData();
            data.append('file_upload_incoming', '1');

            // Add the files
            $.each(self.files, function(key, field)
            {
                $.each(field, function(key, value)
                {
                    data.append(key, value);
                });
            });

            // If there is uploadData passed append it
            $.each(self.settings.uploadData, function(key, value)
            {
                data.append(key, value);
            });

            // Perform Ajax call
            $.ajax($.extend({}, {
                url: self.$form.attr('action'),
                type: 'POST',
                data: data,
                cache: false,
                dataType: 'json',
                processData: false, // Don't process the files, we're using FormData
                contentType: false, // Set content type to false as jQuery will tell the server its a query string request
                success: function(data, textStatus, jqXHR){
                    self.processSubmit(event, data);
                    self.settings.success(data);

                },
                error: function(jqXHR, textStatus, errorThrown){ self.settings.error(jqXHR, textStatus, errorThrown); }
            }, self.settings.uploadOptions));
        },



        /**
         * Submits form data with files
         *
         * @param  jQuery event
         * @param  object
         * @return void
         */
        processSubmit: function(event, uploadData)
        {
            var self = event.data.context;
            // Run the beforeSubmit callback
            if(!self.settings.beforeSubmit(uploadData)) return;

            // Serialize the form data
            var data = self.$form.serializeArray();

            // Loop through the returned array from the server and add it to the new POST
            $.each(uploadData, function(key, value)
            {
                data.push({
                    'name'  : key,
                    'value' : value
                });
            });

            // If there is uploadData passed append it
            $.each(self.settings.submitData, function(key, value)
            {
                data.push({
                    'name'  : key,
                    'value' : value
                });
            });
/*
            $.ajax($.extend({}, {
                url: self.$form.attr('action'),
                type: 'POST',
                data: data,
                cache: false,
                dataType: 'json',
                success: function(data, textStatus, jqXHR){ self.settings.success(data, textStatus, jqXHR); },
                error: function(jqXHR, textStatus, errorThrown){ self.settings.error(jqXHR, textStatus, errorThrown); },
                complete: function(jqXHR, textStatus){ self.settings.complete(jqXHR, textStatus); }
            }, self.settings.submitOptions));*/
        }
    };

    $.fn[pluginName] = function(options)
    {
        return this.each(function()
        {
            if(!$.data(this, "plugin_" + pluginName))
            {
                $.data(this, "plugin_" + pluginName, new Plugin(this, options));
            }
        });
    };

})(jQuery, window, document);