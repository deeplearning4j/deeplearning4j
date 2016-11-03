
package org.deeplearning4j.ui.views.html.training

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object TrainingOverview_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class TrainingOverview extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template1[org.deeplearning4j.ui.api.I18N,play.twirl.api.HtmlFormat.Appendable] {

  /**/
  def apply/*1.2*/(i18n: org.deeplearning4j.ui.api.I18N):play.twirl.api.HtmlFormat.Appendable = {
    _display_ {
      {


Seq[Any](format.raw/*1.40*/("""
"""),format.raw/*2.1*/("""<!DOCTYPE html>
<html lang="en">
    <head>

        <meta charset="utf-8">
        <title>"""),_display_(/*7.17*/i18n/*7.21*/.getMessage("train.pagetitle")),format.raw/*7.51*/("""</title>
            <!-- Start Mobile Specific -->
        <meta name="viewport" content="width=device-width, initial-scale=1">
            <!-- End Mobile Specific -->

        <link id="bootstrap-style" href="/assets/css/bootstrap.min.css" rel="stylesheet">
        <link href="/assets/css/bootstrap-responsive.min.css" rel="stylesheet">
        <link id="base-style" href="/assets/css/style.css" rel="stylesheet">
        <link id="base-style-responsive" href="/assets/css/style-responsive.css" rel="stylesheet">
        <link href='http://fonts.googleapis.com/css?family=Open+Sans:300italic,400italic,600italic,700italic,800italic,400,300,600,700,800&subset=latin,cyrillic-ext,latin-ext' rel='stylesheet' type='text/css'>
        <link rel="shortcut icon" href="/assets/img/favicon.ico">

            <!-- The HTML5 shim, for IE6-8 support of HTML5 elements -->
            <!--[if lt IE 9]>
	  	<script src="http://html5shim.googlecode.com/svn/trunk/html5.js"></script>
		<link id="ie-style" href="/assets/css/ie.css" rel="stylesheet"/>
	<![endif]-->

            <!--[if IE 9]>
		<link id="ie9style" href="/assets/css/ie9.css" rel="stylesheet"/>
	<![endif]-->

    </head>

    <body>
            <!-- Start Header -->
        <div class="navbar">
            <div class="navbar-inner">
                <div class="container-fluid">
                    <a class="btn btn-navbar" data-toggle="collapse" data-target=".top-nav.nav-collapse,.sidebar-nav.nav-collapse">
                        <span class="icon-bar"></span>
                        <span class="icon-bar"></span>
                        <span class="icon-bar"></span>
                    </a>
                    <a class="brand" href="#"><span>"""),_display_(/*41.54*/i18n/*41.58*/.getMessage("train.pagetitle")),format.raw/*41.88*/("""</span></a>
                </div>
            </div>
        </div>
            <!-- End Header -->

        <div class="container-fluid-full">
            <div class="row-fluid">

                    <!-- Start Main Menu -->
                <div id="sidebar-left" class="span2">
                    <div class="nav-collapse sidebar-nav">
                        <ul class="nav nav-tabs nav-stacked main-menu">
                            <li class="active"><a href="javascript:void(0);"><i class="icon-bar-chart"></i><span class="hidden-tablet">"""),_display_(/*54.137*/i18n/*54.141*/.getMessage("train.nav.overview")),format.raw/*54.174*/("""</span></a></li>
                            <li><a href="model"><i class="icon-tasks"></i><span class="hidden-tablet">"""),_display_(/*55.104*/i18n/*55.108*/.getMessage("train.nav.model")),format.raw/*55.138*/("""</span></a></li>
                            <li><a href="system"><i class="icon-dashboard"></i><span class="hidden-tablet">"""),_display_(/*56.109*/i18n/*56.113*/.getMessage("train.nav.system")),format.raw/*56.144*/("""</span></a></li>
                            <li><a href="help"><i class="icon-star"></i><span class="hidden-tablet">"""),_display_(/*57.102*/i18n/*57.106*/.getMessage("train.nav.userguide")),format.raw/*57.140*/("""</span></a></li>
                            <li>
                                <a class="dropmenu" href="javascript:void(0);"><i class="icon-folder-close-alt"></i><span class="hidden-tablet">"""),_display_(/*59.146*/i18n/*59.150*/.getMessage("train.nav.language")),format.raw/*59.183*/("""</span></a>
                                <ul>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('en', 'overview')"><i class="icon-file-alt"></i><span class="hidden-tablet">
                                        English</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ja', 'overview')"><i class="icon-file-alt"></i><span class="hidden-tablet">
                                        日本語</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('zh', 'overview')"><i class="icon-file-alt"></i><span class="hidden-tablet">
                                        中文</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ko', 'overview')"><i class="icon-file-alt"></i><span class="hidden-tablet">
                                        한글</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ru', 'overview')"><i class="icon-file-alt"></i><span class="hidden-tablet">
                                        русский</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('uk', 'overview')"><i class="icon-file-alt"></i><span class="hidden-tablet">
                                        український</span></a></li>
                                </ul>
                            </li>
                        </ul>
                    </div>
                </div>
                    <!-- End Main Menu -->

                <noscript>
                    <div class="alert alert-block span10">
                        <h4 class="alert-heading">Warning!</h4>
                        <p>You need to have <a href="http://en.wikipedia.org/wiki/JavaScript" target="_blank">
                            JavaScript</a> enabled to use this site.</p>
                    </div>
                </noscript>

                    <!-- Start Score Chart-->
                <div id="content" class="span10">

                    <div class="row-fluid">

                        <div class="box span8">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*95.41*/i18n/*95.45*/.getMessage("train.overview.chart.scoreTitle")),format.raw/*95.91*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <div id="scoreiterchart" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*99.55*/i18n/*99.59*/.getMessage("train.overview.chart.scoreTitleShort")),format.raw/*99.110*/("""
                                    """),format.raw/*100.37*/(""":</b> <span id="y">0</span>, <b>"""),_display_(/*100.70*/i18n/*100.74*/.getMessage("train.overview.charts.iteration")),format.raw/*100.120*/("""
                                    """),format.raw/*101.37*/(""":</b> <span id="x">
                                    0</span></p>
                            </div>
                        </div>
                            <!-- End Score Chart-->
                            <!-- Start Model Table-->
                        <div class="box span4">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*109.41*/i18n/*109.45*/.getMessage("train.overview.perftable.title")),format.raw/*109.90*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <table class="table table-bordered table-striped table-condensed">
                                    <tr>
                                        <td>"""),_display_(/*114.46*/i18n/*114.50*/.getMessage("train.overview.modeltable.modeltype")),format.raw/*114.100*/("""</td>
                                        <td id="modelType">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*118.46*/i18n/*118.50*/.getMessage("train.overview.modeltable.nLayers")),format.raw/*118.98*/("""</td>
                                        <td id="nLayers">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*122.46*/i18n/*122.50*/.getMessage("train.overview.modeltable.nParams")),format.raw/*122.98*/("""</td>
                                        <td id="nParams">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*126.46*/i18n/*126.50*/.getMessage("train.overview.perftable.startTime")),format.raw/*126.99*/("""</td>
                                        <td id="startTime">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*130.46*/i18n/*130.50*/.getMessage("train.overview.perftable.totalRuntime")),format.raw/*130.102*/("""</td>
                                        <td id="totalRuntime">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*134.46*/i18n/*134.50*/.getMessage("train.overview.perftable.lastUpdate")),format.raw/*134.100*/("""</td>
                                        <td id="lastUpdate">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*138.46*/i18n/*138.50*/.getMessage("train.overview.perftable.totalParamUpdates")),format.raw/*138.107*/("""</td>
                                        <td id="totalParamUpdates">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*142.46*/i18n/*142.50*/.getMessage("train.overview.perftable.updatesPerSec")),format.raw/*142.103*/("""</td>
                                        <td id="updatesPerSec">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*146.46*/i18n/*146.50*/.getMessage("train.overview.perftable.examplesPerSec")),format.raw/*146.104*/("""</td>
                                        <td id="examplesPerSec">Loading...</td>
                                    </tr>
                                </table>
                            </div>
                        </div>
                            <!--End Model Table -->
                    </div>


                    <div class="row-fluid">
                            <!--Start Ratio Table -->
                        <div class="box span6">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*160.41*/i18n/*160.45*/.getMessage("train.overview.chart.updateRatioTitle")),format.raw/*160.97*/(""": log<sub>10</sub></b></h2>
                            </div>
                            <div class="box-content">
                                <div id="updateRatioChart" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*164.55*/i18n/*164.59*/.getMessage("train.overview.chart.updateRatioTitleShort")),format.raw/*164.116*/("""
                                    """),format.raw/*165.37*/(""":</b> <span id="yRatio">0</span>, <b>log<sub>
                                    10</sub> """),_display_(/*166.47*/i18n/*166.51*/.getMessage("train.overview.chart.updateRatioTitleShort")),format.raw/*166.108*/("""
                                    """),format.raw/*167.37*/(""":</b> <span id="yLogRatio">0</span>
                                    , <b>"""),_display_(/*168.43*/i18n/*168.47*/.getMessage("train.overview.charts.iteration")),format.raw/*168.93*/(""":</b> <span id="xRatio">
                                        0</span></p>
                            </div>
                        </div>
                            <!--End Ratio Table -->
                            <!--Start Variance Table -->
                        <div class="box span6">
                            <div class="box-header">
                                <h2><b><h2><b>"""),_display_(/*176.48*/i18n/*176.52*/.getMessage("train.overview.chart.stdevTitle")),format.raw/*176.98*/(""": log<sub>
                                    10</sub></b></h2></b></h2>
                                <div style="float: right">
                                    <p class="stackControls center">
                                        <input class="btn btn-small" type="button" value=""""),_display_(/*180.92*/i18n/*180.96*/.getMessage("train.overview.chart.stdevBtn.activations")),format.raw/*180.152*/("""" onclick="selectStdevChart('stdevActivations')">
                                        <input class="btn btn-small" type="button" value=""""),_display_(/*181.92*/i18n/*181.96*/.getMessage("train.overview.chart.stdevBtn.gradients")),format.raw/*181.150*/("""" onclick="selectStdevChart('stdevGradients')">
                                        <input class="btn btn-small" type="button" value=""""),_display_(/*182.92*/i18n/*182.96*/.getMessage("train.overview.chart.stdevBtn.updates")),format.raw/*182.148*/("""" onclick="selectStdevChart('stdevUpdates')">
                                    </p>
                                </div>
                            </div>
                            <div class="box-content">
                                <div id="stdevChart" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*188.55*/i18n/*188.59*/.getMessage("train.overview.chart.stdevTitleShort")),format.raw/*188.110*/("""
                                    """),format.raw/*189.37*/(""":</b> <span id="yStdev">0</span>, <b>log<sub>
                                    10</sub> """),_display_(/*190.47*/i18n/*190.51*/.getMessage("train.overview.chart.stdevTitleShort")),format.raw/*190.102*/("""
                                    """),format.raw/*191.37*/(""":</b> <span id="yLogStdev">0</span>
                                    , <b>"""),_display_(/*192.43*/i18n/*192.47*/.getMessage("train.overview.charts.iteration")),format.raw/*192.93*/(""":</b> <span id="xStdev">
                                        0</span></p>
                            </div>
                        </div>
                            <!-- End Variance Table -->
                    </div>
                </div>
            </div><!-- End Content Span10-->
        </div><!--End Row Fluid-->

        <!-- Start JavaScript-->
        <script src="/assets/js/jquery-1.9.1.min.js"></script>
        <script src="/assets/js/jquery-migrate-1.0.0.min.js"></script>
        <script src="/assets/js/jquery-ui-1.10.0.custom.min.js"></script>
        <script src="/assets/js/jquery.ui.touch-punch.js"></script>
        <script src="/assets/js/modernizr.js"></script>
        <script src="/assets/js/bootstrap.min.js"></script>
        <script src="/assets/js/jquery.cookie.js"></script>
        <script src="/assets/js/fullcalendar.min.js"></script>
        <script src="/assets/js/jquery.dataTables.min.js"></script>
        <script src="/assets/js/excanvas.js"></script>
        <script src="/assets/js/jquery.flot.js"></script>
        <script src="/assets/js/jquery.flot.pie.js"></script>
        <script src="/assets/js/jquery.flot.stack.js"></script>
        <script src="/assets/js/jquery.flot.resize.min.js"></script>
        <script src="/assets/js/jquery.chosen.min.js"></script>
        <script src="/assets/js/jquery.uniform.min.js"></script>
        <script src="/assets/js/jquery.cleditor.min.js"></script>
        <script src="/assets/js/jquery.noty.js"></script>
        <script src="/assets/js/jquery.elfinder.min.js"></script>
        <script src="/assets/js/jquery.raty.min.js"></script>
        <script src="/assets/js/jquery.iphone.toggle.js"></script>
        <script src="/assets/js/jquery.uploadify-3.1.min.js"></script>
        <script src="/assets/js/jquery.gritter.min.js"></script>
        <script src="/assets/js/jquery.imagesloaded.js"></script>
        <script src="/assets/js/jquery.masonry.min.js"></script>
        <script src="/assets/js/jquery.knob.modified.js"></script>
        <script src="/assets/js/jquery.sparkline.min.js"></script>
        <script src="/assets/js/counter.js"></script>
        <script src="/assets/js/retina.js"></script>
        <script src="/assets/js/train/overview.js"></script>    <!-- Charts and tables are generated here! -->
        <script src="/assets/js/train/train.js"></script>   <!-- Common (lang selection, etc) -->

        <!-- Execute once on page load -->
        <script>
                $(document).ready(function () """),format.raw/*237.47*/("""{"""),format.raw/*237.48*/("""
                    """),format.raw/*238.21*/("""renderOverviewPage();
                """),format.raw/*239.17*/("""}"""),format.raw/*239.18*/(""");
        </script>

            <!-- Execute periodically (every 2 sec) -->
        <script>
                setInterval(function () """),format.raw/*244.41*/("""{"""),format.raw/*244.42*/("""
                    """),format.raw/*245.21*/("""renderOverviewPage();
                """),format.raw/*246.17*/("""}"""),format.raw/*246.18*/(""", 2000);
        </script>
            <!-- End JavaScript-->

    </body>
</html>
"""))
      }
    }
  }

  def render(i18n:org.deeplearning4j.ui.api.I18N): play.twirl.api.HtmlFormat.Appendable = apply(i18n)

  def f:((org.deeplearning4j.ui.api.I18N) => play.twirl.api.HtmlFormat.Appendable) = (i18n) => apply(i18n)

  def ref: this.type = this

}


}

/**/
object TrainingOverview extends TrainingOverview_Scope0.TrainingOverview
              /*
                  -- GENERATED --
                  DATE: Thu Nov 03 19:29:18 AEDT 2016
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingOverview.scala.html
                  HASH: c535e17458909f245a1f446bef6293f93acd9e18
                  MATRIX: 604->1|737->39|765->41|888->138|900->142|950->172|2726->1921|2739->1925|2790->1955|3379->2516|3393->2520|3448->2553|3597->2674|3611->2678|3663->2708|3817->2834|3831->2838|3884->2869|4031->2988|4045->2992|4101->3026|4326->3223|4340->3227|4395->3260|6872->5710|6885->5714|6952->5760|7239->6020|7252->6024|7325->6075|7392->6113|7453->6146|7467->6150|7536->6196|7603->6234|8021->6624|8035->6628|8102->6673|8419->6962|8433->6966|8506->7016|8747->7229|8761->7233|8831->7281|9070->7492|9084->7496|9154->7544|9393->7755|9407->7759|9478->7808|9719->8021|9733->8025|9808->8077|10052->8293|10066->8297|10139->8347|10381->8561|10395->8565|10475->8622|10724->8843|10738->8847|10814->8900|11059->9117|11073->9121|11150->9175|11747->9744|11761->9748|11835->9800|12143->10080|12157->10084|12237->10141|12304->10179|12425->10272|12439->10276|12519->10333|12586->10371|12693->10450|12707->10454|12775->10500|13212->10909|13226->10913|13294->10959|13619->11256|13633->11260|13712->11316|13882->11458|13896->11462|13973->11516|14141->11656|14155->11660|14230->11712|14632->12086|14646->12090|14720->12141|14787->12179|14908->12272|14922->12276|14996->12327|15063->12365|15170->12444|15184->12448|15252->12494|17853->15066|17883->15067|17934->15089|18002->15128|18032->15129|18201->15269|18231->15270|18282->15292|18350->15331|18380->15332
                  LINES: 20->1|25->1|26->2|31->7|31->7|31->7|65->41|65->41|65->41|78->54|78->54|78->54|79->55|79->55|79->55|80->56|80->56|80->56|81->57|81->57|81->57|83->59|83->59|83->59|119->95|119->95|119->95|123->99|123->99|123->99|124->100|124->100|124->100|124->100|125->101|133->109|133->109|133->109|138->114|138->114|138->114|142->118|142->118|142->118|146->122|146->122|146->122|150->126|150->126|150->126|154->130|154->130|154->130|158->134|158->134|158->134|162->138|162->138|162->138|166->142|166->142|166->142|170->146|170->146|170->146|184->160|184->160|184->160|188->164|188->164|188->164|189->165|190->166|190->166|190->166|191->167|192->168|192->168|192->168|200->176|200->176|200->176|204->180|204->180|204->180|205->181|205->181|205->181|206->182|206->182|206->182|212->188|212->188|212->188|213->189|214->190|214->190|214->190|215->191|216->192|216->192|216->192|261->237|261->237|262->238|263->239|263->239|268->244|268->244|269->245|270->246|270->246
                  -- GENERATED --
              */
          