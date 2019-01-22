
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

<!--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ~ Copyright (c) 2015-2018 Skymind, Inc.
  ~
  ~ This program and the accompanying materials are made available under the
  ~ terms of the Apache License, Version 2.0 which is available at
  ~ https://www.apache.org/licenses/LICENSE-2.0.
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
  ~ WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
  ~ License for the specific language governing permissions and limitations
  ~ under the License.
  ~
  ~ SPDX-License-Identifier: Apache-2.0
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~-->

<html lang="en">
    <head>

        <meta charset="utf-8">
        <title>"""),_display_(/*24.17*/i18n/*24.21*/.getMessage("train.pagetitle")),format.raw/*24.51*/("""</title>
            <!-- Start Mobile Specific -->
        <meta name="viewport" content="width=device-width, initial-scale=1">
            <!-- End Mobile Specific -->

        <link id="bootstrap-style" href="/assets/webjars/bootstrap/2.3.1/css/bootstrap.min.css" rel="stylesheet">
        <link id="bootstrap-style" href="/assets/webjars/bootstrap/2.3.1/css/bootstrap-responsive.min.css" rel="stylesheet">
        <link id="base-style" href="/assets/css/style.css" rel="stylesheet">
        <link id="base-style-responsive" href="/assets/css/style-responsive.css" rel="stylesheet">
        """),format.raw/*33.251*/("""
        """),format.raw/*34.9*/("""<link href='/assets/css/opensans-fonts.css' rel='stylesheet' type='text/css'>
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
                    <a class="brand" href="#"><span>"""),_display_(/*59.54*/i18n/*59.58*/.getMessage("train.pagetitle")),format.raw/*59.88*/("""</span></a>
                    <div id="sessionSelectDiv" style="display:none; float:right">
                        """),_display_(/*61.26*/i18n/*61.30*/.getMessage("train.session.label")),format.raw/*61.64*/("""
                        """),format.raw/*62.25*/("""<select id="sessionSelect" onchange='selectNewSession()'>
                            <option>(Session ID)</option>
                        </select>
                    </div>
                    <div id="workerSelectDiv" style="display:none; float:right;">
                        """),_display_(/*67.26*/i18n/*67.30*/.getMessage("train.session.worker.label")),format.raw/*67.71*/("""
                        """),format.raw/*68.25*/("""<select id="workerSelect" onchange='selectNewWorker()'>
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
                            <li class="active"><a href="javascript:void(0);"><i class="icon-bar-chart"></i><span class="hidden-tablet">"""),_display_(/*84.137*/i18n/*84.141*/.getMessage("train.nav.overview")),format.raw/*84.174*/("""</span></a></li>
                            <li><a href="model"><i class="icon-tasks"></i><span class="hidden-tablet">"""),_display_(/*85.104*/i18n/*85.108*/.getMessage("train.nav.model")),format.raw/*85.138*/("""</span></a></li>
                            <li><a href="system"><i class="icon-dashboard"></i><span class="hidden-tablet">"""),_display_(/*86.109*/i18n/*86.113*/.getMessage("train.nav.system")),format.raw/*86.144*/("""</span></a></li>
                            """),format.raw/*87.160*/("""
                            """),format.raw/*88.29*/("""<li>
                                <a class="dropmenu" href="javascript:void(0);"><i class="icon-folder-close-alt"></i><span class="hidden-tablet">"""),_display_(/*89.146*/i18n/*89.150*/.getMessage("train.nav.language")),format.raw/*89.183*/("""</span></a>
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
                                <h2><b>"""),_display_(/*127.41*/i18n/*127.45*/.getMessage("train.overview.chart.scoreTitle")),format.raw/*127.91*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <div id="scoreiterchart" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*131.55*/i18n/*131.59*/.getMessage("train.overview.chart.scoreTitleShort")),format.raw/*131.110*/("""
                                    """),format.raw/*132.37*/(""":</b> <span id="y">0</span>, <b>"""),_display_(/*132.70*/i18n/*132.74*/.getMessage("train.overview.charts.iteration")),format.raw/*132.120*/("""
                                    """),format.raw/*133.37*/(""":</b> <span id="x">
                                    0</span></p>
                            </div>
                        </div>
                            <!-- End Score Chart-->
                            <!-- Start Model Table-->
                        <div class="box span4">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*141.41*/i18n/*141.45*/.getMessage("train.overview.perftable.title")),format.raw/*141.90*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <table class="table table-bordered table-striped table-condensed">
                                    <tr>
                                        <td>"""),_display_(/*146.46*/i18n/*146.50*/.getMessage("train.overview.modeltable.modeltype")),format.raw/*146.100*/("""</td>
                                        <td id="modelType">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*150.46*/i18n/*150.50*/.getMessage("train.overview.modeltable.nLayers")),format.raw/*150.98*/("""</td>
                                        <td id="nLayers">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*154.46*/i18n/*154.50*/.getMessage("train.overview.modeltable.nParams")),format.raw/*154.98*/("""</td>
                                        <td id="nParams">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*158.46*/i18n/*158.50*/.getMessage("train.overview.perftable.startTime")),format.raw/*158.99*/("""</td>
                                        <td id="startTime">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*162.46*/i18n/*162.50*/.getMessage("train.overview.perftable.totalRuntime")),format.raw/*162.102*/("""</td>
                                        <td id="totalRuntime">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*166.46*/i18n/*166.50*/.getMessage("train.overview.perftable.lastUpdate")),format.raw/*166.100*/("""</td>
                                        <td id="lastUpdate">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*170.46*/i18n/*170.50*/.getMessage("train.overview.perftable.totalParamUpdates")),format.raw/*170.107*/("""</td>
                                        <td id="totalParamUpdates">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*174.46*/i18n/*174.50*/.getMessage("train.overview.perftable.updatesPerSec")),format.raw/*174.103*/("""</td>
                                        <td id="updatesPerSec">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*178.46*/i18n/*178.50*/.getMessage("train.overview.perftable.examplesPerSec")),format.raw/*178.104*/("""</td>
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
                                <h2><b>"""),_display_(/*192.41*/i18n/*192.45*/.getMessage("train.overview.chart.updateRatioTitle")),format.raw/*192.97*/(""": log<sub>10</sub></b></h2>
                            </div>
                            <div class="box-content">
                                <div id="updateRatioChart" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*196.55*/i18n/*196.59*/.getMessage("train.overview.chart.updateRatioTitleShort")),format.raw/*196.116*/("""
                                    """),format.raw/*197.37*/(""":</b> <span id="yRatio">0</span>, <b>log<sub>
                                    10</sub> """),_display_(/*198.47*/i18n/*198.51*/.getMessage("train.overview.chart.updateRatioTitleShort")),format.raw/*198.108*/("""
                                    """),format.raw/*199.37*/(""":</b> <span id="yLogRatio">0</span>
                                    , <b>"""),_display_(/*200.43*/i18n/*200.47*/.getMessage("train.overview.charts.iteration")),format.raw/*200.93*/(""":</b> <span id="xRatio">
                                        0</span></p>
                            </div>
                        </div>
                            <!--End Ratio Table -->
                            <!--Start Variance Table -->
                        <div class="box span6">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*208.41*/i18n/*208.45*/.getMessage("train.overview.chart.stdevTitle")),format.raw/*208.91*/(""": log<sub>10</sub></b></h2>
                                <ul class="nav tab-menu nav-tabs" style="position:absolute; margin-top: -11px; right: 22px;">
                                    <li class="active" id="stdevActivations"><a href="javascript:void(0);" onclick="selectStdevChart('stdevActivations')">"""),_display_(/*210.156*/i18n/*210.160*/.getMessage("train.overview.chart.stdevBtn.activations")),format.raw/*210.216*/("""</a></li>
                                    <li id="stdevGradients"><a href="javascript:void(0);" onclick="selectStdevChart('stdevGradients')">"""),_display_(/*211.137*/i18n/*211.141*/.getMessage("train.overview.chart.stdevBtn.gradients")),format.raw/*211.195*/("""</a></li>
                                    <li id="stdevUpdates"><a href="javascript:void(0);" onclick="selectStdevChart('stdevUpdates')">"""),_display_(/*212.133*/i18n/*212.137*/.getMessage("train.overview.chart.stdevBtn.updates")),format.raw/*212.189*/("""</a></li>
                                </ul>
                            </div>
                            <div class="box-content">
                                <div id="stdevChart" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*217.55*/i18n/*217.59*/.getMessage("train.overview.chart.stdevTitleShort")),format.raw/*217.110*/("""
                                    """),format.raw/*218.37*/(""":</b> <span id="yStdev">0</span>, <b>log<sub>
                                    10</sub> """),_display_(/*219.47*/i18n/*219.51*/.getMessage("train.overview.chart.stdevTitleShort")),format.raw/*219.102*/("""
                                    """),format.raw/*220.37*/(""":</b> <span id="yLogStdev">0</span>
                                    , <b>"""),_display_(/*221.43*/i18n/*221.47*/.getMessage("train.overview.charts.iteration")),format.raw/*221.93*/(""":</b> <span id="xStdev">
                                        0</span></p>
                            </div>
                        </div>
                            <!-- End Variance Table -->
                    </div>
                </div>
            </div><!-- End Content Span10-->
        </div><!--End Row Fluid-->

        <!-- Start JavaScript-->
        <script src="/assets/webjars/jquery/2.2.0/jquery.min.js"></script>
        <script src="/assets/webjars/jquery-ui/1.10.2/ui/minified/jquery-ui.min.js"></script>
        <script src="/assets/webjars/jquery-migrate/1.2.1/jquery-migrate.min.js"></script>
        <script src="/assets/webjars/modernizr/2.8.3/modernizr.min.js"></script>
        <script src="/assets/webjars/bootstrap/2.3.1/js/bootstrap.min.js"></script>
        <script src="/assets/webjars/jquery-cookie/1.4.1-1/jquery.cookie.js"></script>
        <script src="/assets/webjars/fullcalendar/1.6.4/fullcalendar.min.js"></script>
        <script src="/assets/webjars/excanvas/3/excanvas.js"></script>
        <script src="/assets/webjars/retinajs/0.0.2/retina.js"></script>
        <script src="/assets/webjars/flot/0.8.3/jquery.flot.js"></script>
        <script src="/assets/webjars/flot/0.8.3/jquery.flot.pie.js"></script>
        <script src="/assets/webjars/flot/0.8.3/jquery.flot.stack.js"></script>
        <script src="/assets/webjars/flot/0.8.3/jquery.flot.resize.min.js"></script>
        <script src="/assets/webjars/flot/0.8.3/jquery.flot.selection.js"></script>
        <script src="/assets/webjars/chosen/0.9.8/chosen/chosen.jquery.min.js"></script>
        <script src="/assets/webjars/uniform/2.1.2/jquery.uniform.min.js"></script>
        <script src="/assets/webjars/noty/2.2.2/jquery.noty.packaged.js"></script>
        <script src="/assets/webjars/jquery-raty/2.5.2/jquery.raty.min.js"></script>
        <script src="/assets/webjars/imagesloaded/2.1.1/jquery.imagesloaded.min.js"></script>
        <script src="/assets/webjars/masonry/3.1.5/masonry.pkgd.min.js"></script>
        <script src="/assets/webjars/jquery.sparkline/2.1.2/jquery.sparkline.min.js"></script>
        <script src="/assets/webjars/jquery-knob/1.2.2/jquery.knob.min.js"></script>
        <script src="/assets/webjars/datatables/1.9.4/media/js/jquery.dataTables.min.js"></script>
        <script src="/assets/webjars/jquery-ui-touch-punch/0.2.2/jquery.ui.touch-punch.min.js"></script>
        <script src="/assets/webjars/github-com-jboesch-Gritter/1.7.4/jquery.gritter.js"></script>

        <script src="/assets/js/train/overview.js"></script>    <!-- Charts and tables are generated here! -->
        <script src="/assets/js/train/train.js"></script>   <!-- Common (lang selection, etc) -->
        <script src="/assets/js/counter.js"></script>

        <!-- Execute once on page load -->
        <script>
                $(document).ready(function () """),format.raw/*264.47*/("""{"""),format.raw/*264.48*/("""
                    """),format.raw/*265.21*/("""renderOverviewPage(true);
                """),format.raw/*266.17*/("""}"""),format.raw/*266.18*/(""");
        </script>

            <!-- Execute periodically (every 2 sec) -->
        <script>
                setInterval(function () """),format.raw/*271.41*/("""{"""),format.raw/*271.42*/("""
                    """),format.raw/*272.21*/("""renderOverviewPage(false);
                """),format.raw/*273.17*/("""}"""),format.raw/*273.18*/(""", 2000);
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
                  DATE: Tue Jan 22 16:25:54 AEDT 2019
                  SOURCE: c:/DL4J/Git/deeplearning4j/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingOverview.scala.html
                  HASH: 33a18cbf264fc10b63629e05cd35284b4b3a8189
                  MATRIX: 604->1|737->39|765->41|1683->932|1696->936|1747->966|2379->1811|2416->1821|3534->2912|3547->2916|3598->2946|3746->3067|3759->3071|3814->3105|3868->3131|4184->3420|4197->3424|4259->3465|4313->3491|5067->4217|5081->4221|5136->4254|5285->4375|5299->4379|5351->4409|5505->4535|5519->4539|5572->4570|5647->4747|5705->4777|5884->4928|5898->4932|5953->4965|8685->7669|8699->7673|8767->7719|9055->7979|9069->7983|9143->8034|9210->8072|9271->8105|9285->8109|9354->8155|9421->8193|9839->8583|9853->8587|9920->8632|10237->8921|10251->8925|10324->8975|10565->9188|10579->9192|10649->9240|10888->9451|10902->9455|10972->9503|11211->9714|11225->9718|11296->9767|11537->9980|11551->9984|11626->10036|11870->10252|11884->10256|11957->10306|12199->10520|12213->10524|12293->10581|12542->10802|12556->10806|12632->10859|12877->11076|12891->11080|12968->11134|13565->11703|13579->11707|13653->11759|13961->12039|13975->12043|14055->12100|14122->12138|14243->12231|14257->12235|14337->12292|14404->12330|14511->12409|14525->12413|14593->12459|15023->12861|15037->12865|15105->12911|15445->13222|15460->13226|15539->13282|15715->13429|15730->13433|15807->13487|15979->13630|15994->13634|16069->13686|16392->13981|16406->13985|16480->14036|16547->14074|16668->14167|16682->14171|16756->14222|16823->14260|16930->14339|16944->14343|17012->14389|19963->17311|19993->17312|20044->17334|20116->17377|20146->17378|20315->17518|20345->17519|20396->17541|20469->17585|20499->17586
                  LINES: 20->1|25->1|26->2|48->24|48->24|48->24|57->33|58->34|83->59|83->59|83->59|85->61|85->61|85->61|86->62|91->67|91->67|91->67|92->68|108->84|108->84|108->84|109->85|109->85|109->85|110->86|110->86|110->86|111->87|112->88|113->89|113->89|113->89|151->127|151->127|151->127|155->131|155->131|155->131|156->132|156->132|156->132|156->132|157->133|165->141|165->141|165->141|170->146|170->146|170->146|174->150|174->150|174->150|178->154|178->154|178->154|182->158|182->158|182->158|186->162|186->162|186->162|190->166|190->166|190->166|194->170|194->170|194->170|198->174|198->174|198->174|202->178|202->178|202->178|216->192|216->192|216->192|220->196|220->196|220->196|221->197|222->198|222->198|222->198|223->199|224->200|224->200|224->200|232->208|232->208|232->208|234->210|234->210|234->210|235->211|235->211|235->211|236->212|236->212|236->212|241->217|241->217|241->217|242->218|243->219|243->219|243->219|244->220|245->221|245->221|245->221|288->264|288->264|289->265|290->266|290->266|295->271|295->271|296->272|297->273|297->273
                  -- GENERATED --
              */
          