function onNavClick(source, errorMsg) {
    console.log("Clicked: " + source);
    var reqURL = "train/" + source;
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
            $.notify(errorMsg, "error");
        }
    });
}
function changeLanguage(errorMsg) {
    var currSelected = $("#navLangSelect").val();
    $.ajax({
        url: "setlang/" + currSelected,
        success: function () {
            location.reload(true);
            console.log("Selected language: " + currSelected);
        },
        error: function (query, status, error) {
            $.notify(errorMsg, "error");
        }
    });
}
function setLanguageSelectorValue() {
    $.ajax({
        url: "/lang/getCurrent",
        success: function (data) {
            $("#navLangSelect").val(data);
        }
    });
}
//# sourceMappingURL=dl4j-play-ui.js.map