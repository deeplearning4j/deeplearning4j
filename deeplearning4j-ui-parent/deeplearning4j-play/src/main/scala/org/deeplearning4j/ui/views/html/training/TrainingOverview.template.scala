
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
                    <div id="sessionSelectDiv" style="display:none; float:right">
                        """),_display_(/*43.26*/i18n/*43.30*/.getMessage("train.session.label")),format.raw/*43.64*/("""
                        """),format.raw/*44.25*/("""<select id="sessionSelect" onchange='selectNewSession()'>
                            <option>(Session ID)</option>
                        </select>
                    </div>
                    <div id="workerSelectDiv" style="display:none; float:right;">
                        """),_display_(/*49.26*/i18n/*49.30*/.getMessage("train.session.worker.label")),format.raw/*49.71*/("""
                        """),format.raw/*50.25*/("""<select id="workerSelect" onchange='selectNewWorker()'>
                            <option>(Worker ID)</option>
                        </select>
                    </div>
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
                            <li class="active"><a href="javascript:void(0);"><i class="icon-bar-chart"></i><span class="hidden-tablet">"""),_display_(/*66.137*/i18n/*66.141*/.getMessage("train.nav.overview")),format.raw/*66.174*/("""</span></a></li>
                            <li><a href="model"><i class="icon-tasks"></i><span class="hidden-tablet">"""),_display_(/*67.104*/i18n/*67.108*/.getMessage("train.nav.model")),format.raw/*67.138*/("""</span></a></li>
                            <li><a href="system"><i class="icon-dashboard"></i><span class="hidden-tablet">"""),_display_(/*68.109*/i18n/*68.113*/.getMessage("train.nav.system")),format.raw/*68.144*/("""</span></a></li>
                            """),format.raw/*69.160*/("""
                            """),format.raw/*70.29*/("""<li>
                                <a class="dropmenu" href="javascript:void(0);"><i class="icon-folder-close-alt"></i><span class="hidden-tablet">"""),_display_(/*71.146*/i18n/*71.150*/.getMessage("train.nav.language")),format.raw/*71.183*/("""</span></a>
                                <ul>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('en', 'overview')"><i class="icon-file-alt"></i><span class="hidden-tablet">
                                        English</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('de', 'overview')"><i class="icon-file-alt"></i> <span class="hidden-tablet">
                                        Deutsch</span></a></li>
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
                                <h2><b>"""),_display_(/*109.41*/i18n/*109.45*/.getMessage("train.overview.chart.scoreTitle")),format.raw/*109.91*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <div id="scoreiterchart" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*113.55*/i18n/*113.59*/.getMessage("train.overview.chart.scoreTitleShort")),format.raw/*113.110*/("""
                                    """),format.raw/*114.37*/(""":</b> <span id="y">0</span>, <b>"""),_display_(/*114.70*/i18n/*114.74*/.getMessage("train.overview.charts.iteration")),format.raw/*114.120*/("""
                                    """),format.raw/*115.37*/(""":</b> <span id="x">
                                    0</span></p>
                            </div>
                        </div>
                            <!-- End Score Chart-->
                            <!-- Start Model Table-->
                        <div class="box span4">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*123.41*/i18n/*123.45*/.getMessage("train.overview.perftable.title")),format.raw/*123.90*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <table class="table table-bordered table-striped table-condensed">
                                    <tr>
                                        <td>"""),_display_(/*128.46*/i18n/*128.50*/.getMessage("train.overview.modeltable.modeltype")),format.raw/*128.100*/("""</td>
                                        <td id="modelType">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*132.46*/i18n/*132.50*/.getMessage("train.overview.modeltable.nLayers")),format.raw/*132.98*/("""</td>
                                        <td id="nLayers">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*136.46*/i18n/*136.50*/.getMessage("train.overview.modeltable.nParams")),format.raw/*136.98*/("""</td>
                                        <td id="nParams">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*140.46*/i18n/*140.50*/.getMessage("train.overview.perftable.startTime")),format.raw/*140.99*/("""</td>
                                        <td id="startTime">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*144.46*/i18n/*144.50*/.getMessage("train.overview.perftable.totalRuntime")),format.raw/*144.102*/("""</td>
                                        <td id="totalRuntime">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*148.46*/i18n/*148.50*/.getMessage("train.overview.perftable.lastUpdate")),format.raw/*148.100*/("""</td>
                                        <td id="lastUpdate">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*152.46*/i18n/*152.50*/.getMessage("train.overview.perftable.totalParamUpdates")),format.raw/*152.107*/("""</td>
                                        <td id="totalParamUpdates">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*156.46*/i18n/*156.50*/.getMessage("train.overview.perftable.updatesPerSec")),format.raw/*156.103*/("""</td>
                                        <td id="updatesPerSec">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*160.46*/i18n/*160.50*/.getMessage("train.overview.perftable.examplesPerSec")),format.raw/*160.104*/("""</td>
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
                                <h2><b>"""),_display_(/*174.41*/i18n/*174.45*/.getMessage("train.overview.chart.updateRatioTitle")),format.raw/*174.97*/(""": log<sub>10</sub></b></h2>
                            </div>
                            <div class="box-content">
                                <div id="updateRatioChart" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*178.55*/i18n/*178.59*/.getMessage("train.overview.chart.updateRatioTitleShort")),format.raw/*178.116*/("""
                                    """),format.raw/*179.37*/(""":</b> <span id="yRatio">0</span>, <b>log<sub>
                                    10</sub> """),_display_(/*180.47*/i18n/*180.51*/.getMessage("train.overview.chart.updateRatioTitleShort")),format.raw/*180.108*/("""
                                    """),format.raw/*181.37*/(""":</b> <span id="yLogRatio">0</span>
                                    , <b>"""),_display_(/*182.43*/i18n/*182.47*/.getMessage("train.overview.charts.iteration")),format.raw/*182.93*/(""":</b> <span id="xRatio">
                                        0</span></p>
                            </div>
                        </div>
                            <!--End Ratio Table -->
                            <!--Start Variance Table -->
                        <div class="box span6">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*190.41*/i18n/*190.45*/.getMessage("train.overview.chart.stdevTitle")),format.raw/*190.91*/(""": log<sub>10</sub></b></h2>
                                <ul class="nav tab-menu nav-tabs" style="position:absolute; margin-top: -11px; right: 22px;">
                                    <li class="active" id="stdevActivations"><a href="javascript:void(0);" onclick="selectStdevChart('stdevActivations')">"""),_display_(/*192.156*/i18n/*192.160*/.getMessage("train.overview.chart.stdevBtn.activations")),format.raw/*192.216*/("""</a></li>
                                    <li id="stdevGradients"><a href="javascript:void(0);" onclick="selectStdevChart('stdevGradients')">"""),_display_(/*193.137*/i18n/*193.141*/.getMessage("train.overview.chart.stdevBtn.gradients")),format.raw/*193.195*/("""</a></li>
                                    <li id="stdevUpdates"><a href="javascript:void(0);" onclick="selectStdevChart('stdevUpdates')">"""),_display_(/*194.133*/i18n/*194.137*/.getMessage("train.overview.chart.stdevBtn.updates")),format.raw/*194.189*/("""</a></li>
                                </ul>
                            </div>
                            <div class="box-content">
                                <div id="stdevChart" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*199.55*/i18n/*199.59*/.getMessage("train.overview.chart.stdevTitleShort")),format.raw/*199.110*/("""
                                    """),format.raw/*200.37*/(""":</b> <span id="yStdev">0</span>, <b>log<sub>
                                    10</sub> """),_display_(/*201.47*/i18n/*201.51*/.getMessage("train.overview.chart.stdevTitleShort")),format.raw/*201.102*/("""
                                    """),format.raw/*202.37*/(""":</b> <span id="yLogStdev">0</span>
                                    , <b>"""),_display_(/*203.43*/i18n/*203.47*/.getMessage("train.overview.charts.iteration")),format.raw/*203.93*/(""":</b> <span id="xStdev">
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
        <script src="/assets/js/jquery.flot.selection.js"></script>
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
                $(document).ready(function () """),format.raw/*249.47*/("""{"""),format.raw/*249.48*/("""
                    """),format.raw/*250.21*/("""renderOverviewPage(true);
                """),format.raw/*251.17*/("""}"""),format.raw/*251.18*/(""");
        </script>

            <!-- Execute periodically (every 2 sec) -->
        <script>
                setInterval(function () """),format.raw/*256.41*/("""{"""),format.raw/*256.42*/("""
                    """),format.raw/*257.21*/("""renderOverviewPage(false);
                """),format.raw/*258.17*/("""}"""),format.raw/*258.18*/(""", 2000);
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
                  DATE: Sun Jan 08 12:25:14 AEDT 2017
                  SOURCE: c:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingOverview.scala.html
                  HASH: 778810c8e0df5337a0e6825b5198f72367decea6
                  MATRIX: 604->1|737->39|765->41|888->138|900->142|950->172|2726->1921|2739->1925|2790->1955|2938->2076|2951->2080|3006->2114|3060->2140|3376->2429|3389->2433|3451->2474|3505->2500|4259->3226|4273->3230|4328->3263|4477->3384|4491->3388|4543->3418|4697->3544|4711->3548|4764->3579|4839->3756|4897->3786|5076->3937|5090->3941|5145->3974|7877->6678|7891->6682|7959->6728|8247->6988|8261->6992|8335->7043|8402->7081|8463->7114|8477->7118|8546->7164|8613->7202|9031->7592|9045->7596|9112->7641|9429->7930|9443->7934|9516->7984|9757->8197|9771->8201|9841->8249|10080->8460|10094->8464|10164->8512|10403->8723|10417->8727|10488->8776|10729->8989|10743->8993|10818->9045|11062->9261|11076->9265|11149->9315|11391->9529|11405->9533|11485->9590|11734->9811|11748->9815|11824->9868|12069->10085|12083->10089|12160->10143|12757->10712|12771->10716|12845->10768|13153->11048|13167->11052|13247->11109|13314->11147|13435->11240|13449->11244|13529->11301|13596->11339|13703->11418|13717->11422|13785->11468|14215->11870|14229->11874|14297->11920|14637->12231|14652->12235|14731->12291|14907->12438|14922->12442|14999->12496|15171->12639|15186->12643|15261->12695|15584->12990|15598->12994|15672->13045|15739->13083|15860->13176|15874->13180|15948->13231|16015->13269|16122->13348|16136->13352|16204->13398|18874->16039|18904->16040|18955->16062|19027->16105|19057->16106|19226->16246|19256->16247|19307->16269|19380->16313|19410->16314
                  LINES: 20->1|25->1|26->2|31->7|31->7|31->7|65->41|65->41|65->41|67->43|67->43|67->43|68->44|73->49|73->49|73->49|74->50|90->66|90->66|90->66|91->67|91->67|91->67|92->68|92->68|92->68|93->69|94->70|95->71|95->71|95->71|133->109|133->109|133->109|137->113|137->113|137->113|138->114|138->114|138->114|138->114|139->115|147->123|147->123|147->123|152->128|152->128|152->128|156->132|156->132|156->132|160->136|160->136|160->136|164->140|164->140|164->140|168->144|168->144|168->144|172->148|172->148|172->148|176->152|176->152|176->152|180->156|180->156|180->156|184->160|184->160|184->160|198->174|198->174|198->174|202->178|202->178|202->178|203->179|204->180|204->180|204->180|205->181|206->182|206->182|206->182|214->190|214->190|214->190|216->192|216->192|216->192|217->193|217->193|217->193|218->194|218->194|218->194|223->199|223->199|223->199|224->200|225->201|225->201|225->201|226->202|227->203|227->203|227->203|273->249|273->249|274->250|275->251|275->251|280->256|280->256|281->257|282->258|282->258
                  -- GENERATED --
              */
          