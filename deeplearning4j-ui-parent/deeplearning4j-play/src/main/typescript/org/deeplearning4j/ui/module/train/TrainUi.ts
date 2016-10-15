/// <reference path="../../typedefs/jquery.d.ts" />
/// <reference path="../../typedefs/notify.d.ts" />

function onNavClick(source: String, errorMsg: String) {

    console.log("Clicked: " + source);

    var reqURL: string = "train/" + source;

    $.ajax({
        url: reqURL,
        success: function (data) {
            $("#homeNavDiv").removeClass("navElementSelected");
            $("#modelNavDiv").removeClass("navElementSelected");
            $("#systemNavDiv").removeClass("navElementSelected");
            $("#helpNavDiv").removeClass("navElementSelected");
            $("#" + source + "NavDiv").addClass("navElementSelected");
            $("#mainContentDiv").html(data);
        },
        error: function (query, status, error) {
            $.notify(errorMsg,"error");
        }
    });
}

function changeLanguage(errorMsg: string){
    var currSelected = $("#navLangSelect").val();

    $.ajax({
        url: "setlang/" + currSelected,
        success: function () {
            location.reload(true);
            console.log("Selected language: " + currSelected);
        },
        error: function (query, status, error) {
            $.notify(errorMsg,"error");
        }
    });
}

function setLanguageSelectorValue(){
    $.ajax({
        url: "/lang/getCurrent",
        success: function(data){
            $("#navLangSelect").val(data);
        }
    });
}