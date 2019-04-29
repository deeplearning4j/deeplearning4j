
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

        <link id="bootstrap-style" href="/assets/webjars/bootstrap/2.3.2/css/bootstrap.min.css" rel="stylesheet">
        <link id="bootstrap-style" href="/assets/webjars/bootstrap/2.3.2/css/bootstrap-responsive.min.css" rel="stylesheet">
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
                                <h2><b>"""),_display_(/*125.41*/i18n/*125.45*/.getMessage("train.overview.chart.scoreTitle")),format.raw/*125.91*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <div id="scoreiterchart" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*129.55*/i18n/*129.59*/.getMessage("train.overview.chart.scoreTitleShort")),format.raw/*129.110*/("""
                                    """),format.raw/*130.37*/(""":</b> <span id="y">0</span>, <b>"""),_display_(/*130.70*/i18n/*130.74*/.getMessage("train.overview.charts.iteration")),format.raw/*130.120*/("""
                                    """),format.raw/*131.37*/(""":</b> <span id="x">
                                    0</span></p>
                            </div>
                        </div>
                            <!-- End Score Chart-->
                            <!-- Start Model Table-->
                        <div class="box span4">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*139.41*/i18n/*139.45*/.getMessage("train.overview.perftable.title")),format.raw/*139.90*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <table class="table table-bordered table-striped table-condensed">
                                    <tr>
                                        <td>"""),_display_(/*144.46*/i18n/*144.50*/.getMessage("train.overview.modeltable.modeltype")),format.raw/*144.100*/("""</td>
                                        <td id="modelType">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*148.46*/i18n/*148.50*/.getMessage("train.overview.modeltable.nLayers")),format.raw/*148.98*/("""</td>
                                        <td id="nLayers">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*152.46*/i18n/*152.50*/.getMessage("train.overview.modeltable.nParams")),format.raw/*152.98*/("""</td>
                                        <td id="nParams">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*156.46*/i18n/*156.50*/.getMessage("train.overview.perftable.startTime")),format.raw/*156.99*/("""</td>
                                        <td id="startTime">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*160.46*/i18n/*160.50*/.getMessage("train.overview.perftable.totalRuntime")),format.raw/*160.102*/("""</td>
                                        <td id="totalRuntime">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*164.46*/i18n/*164.50*/.getMessage("train.overview.perftable.lastUpdate")),format.raw/*164.100*/("""</td>
                                        <td id="lastUpdate">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*168.46*/i18n/*168.50*/.getMessage("train.overview.perftable.totalParamUpdates")),format.raw/*168.107*/("""</td>
                                        <td id="totalParamUpdates">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*172.46*/i18n/*172.50*/.getMessage("train.overview.perftable.updatesPerSec")),format.raw/*172.103*/("""</td>
                                        <td id="updatesPerSec">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*176.46*/i18n/*176.50*/.getMessage("train.overview.perftable.examplesPerSec")),format.raw/*176.104*/("""</td>
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
                                <h2><b>"""),_display_(/*190.41*/i18n/*190.45*/.getMessage("train.overview.chart.updateRatioTitle")),format.raw/*190.97*/(""": log<sub>10</sub></b></h2>
                            </div>
                            <div class="box-content">
                                <div id="updateRatioChart" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*194.55*/i18n/*194.59*/.getMessage("train.overview.chart.updateRatioTitleShort")),format.raw/*194.116*/("""
                                    """),format.raw/*195.37*/(""":</b> <span id="yRatio">0</span>, <b>log<sub>
                                    10</sub> """),_display_(/*196.47*/i18n/*196.51*/.getMessage("train.overview.chart.updateRatioTitleShort")),format.raw/*196.108*/("""
                                    """),format.raw/*197.37*/(""":</b> <span id="yLogRatio">0</span>
                                    , <b>"""),_display_(/*198.43*/i18n/*198.47*/.getMessage("train.overview.charts.iteration")),format.raw/*198.93*/(""":</b> <span id="xRatio">
                                        0</span></p>
                            </div>
                        </div>
                            <!--End Ratio Table -->
                            <!--Start Variance Table -->
                        <div class="box span6">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*206.41*/i18n/*206.45*/.getMessage("train.overview.chart.stdevTitle")),format.raw/*206.91*/(""": log<sub>10</sub></b></h2>
                                <ul class="nav tab-menu nav-tabs" style="position:absolute; margin-top: -11px; right: 22px;">
                                    <li class="active" id="stdevActivations"><a href="javascript:void(0);" onclick="selectStdevChart('stdevActivations')">"""),_display_(/*208.156*/i18n/*208.160*/.getMessage("train.overview.chart.stdevBtn.activations")),format.raw/*208.216*/("""</a></li>
                                    <li id="stdevGradients"><a href="javascript:void(0);" onclick="selectStdevChart('stdevGradients')">"""),_display_(/*209.137*/i18n/*209.141*/.getMessage("train.overview.chart.stdevBtn.gradients")),format.raw/*209.195*/("""</a></li>
                                    <li id="stdevUpdates"><a href="javascript:void(0);" onclick="selectStdevChart('stdevUpdates')">"""),_display_(/*210.133*/i18n/*210.137*/.getMessage("train.overview.chart.stdevBtn.updates")),format.raw/*210.189*/("""</a></li>
                                </ul>
                            </div>
                            <div class="box-content">
                                <div id="stdevChart" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*215.55*/i18n/*215.59*/.getMessage("train.overview.chart.stdevTitleShort")),format.raw/*215.110*/("""
                                    """),format.raw/*216.37*/(""":</b> <span id="yStdev">0</span>, <b>log<sub>
                                    10</sub> """),_display_(/*217.47*/i18n/*217.51*/.getMessage("train.overview.chart.stdevTitleShort")),format.raw/*217.102*/("""
                                    """),format.raw/*218.37*/(""":</b> <span id="yLogStdev">0</span>
                                    , <b>"""),_display_(/*219.43*/i18n/*219.47*/.getMessage("train.overview.charts.iteration")),format.raw/*219.93*/(""":</b> <span id="xStdev">
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
        <script src="/assets/webjars/bootstrap/2.3.2/js/bootstrap.min.js"></script>
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
                $(document).ready(function () """),format.raw/*262.47*/("""{"""),format.raw/*262.48*/("""
                    """),format.raw/*263.21*/("""renderOverviewPage(true);
                """),format.raw/*264.17*/("""}"""),format.raw/*264.18*/(""");
        </script>

            <!-- Execute periodically (every 2 sec) -->
        <script>
                setInterval(function () """),format.raw/*269.41*/("""{"""),format.raw/*269.42*/("""
                    """),format.raw/*270.21*/("""renderOverviewPage(false);
                """),format.raw/*271.17*/("""}"""),format.raw/*271.18*/(""", 2000);
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
                  DATE: Wed Mar 13 15:34:55 AEDT 2019
                  SOURCE: c:/DL4J/Git/deeplearning4j/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingOverview.scala.html
                  HASH: 1de0097ec528376537eead526e6aa8e4e1a7c4db
                  MATRIX: 604->1|737->39|765->41|1683->932|1696->936|1747->966|2379->1811|2416->1821|3534->2912|3547->2916|3598->2946|3746->3067|3759->3071|3814->3105|3868->3131|4184->3420|4197->3424|4259->3465|4313->3491|5067->4217|5081->4221|5136->4254|5285->4375|5299->4379|5351->4409|5505->4535|5519->4539|5572->4570|5647->4747|5705->4777|5884->4928|5898->4932|5953->4965|8428->7412|8442->7416|8510->7462|8798->7722|8812->7726|8886->7777|8953->7815|9014->7848|9028->7852|9097->7898|9164->7936|9582->8326|9596->8330|9663->8375|9980->8664|9994->8668|10067->8718|10308->8931|10322->8935|10392->8983|10631->9194|10645->9198|10715->9246|10954->9457|10968->9461|11039->9510|11280->9723|11294->9727|11369->9779|11613->9995|11627->9999|11700->10049|11942->10263|11956->10267|12036->10324|12285->10545|12299->10549|12375->10602|12620->10819|12634->10823|12711->10877|13308->11446|13322->11450|13396->11502|13704->11782|13718->11786|13798->11843|13865->11881|13986->11974|14000->11978|14080->12035|14147->12073|14254->12152|14268->12156|14336->12202|14766->12604|14780->12608|14848->12654|15188->12965|15203->12969|15282->13025|15458->13172|15473->13176|15550->13230|15722->13373|15737->13377|15812->13429|16135->13724|16149->13728|16223->13779|16290->13817|16411->13910|16425->13914|16499->13965|16566->14003|16673->14082|16687->14086|16755->14132|19706->17054|19736->17055|19787->17077|19859->17120|19889->17121|20058->17261|20088->17262|20139->17284|20212->17328|20242->17329
                  LINES: 20->1|25->1|26->2|48->24|48->24|48->24|57->33|58->34|83->59|83->59|83->59|85->61|85->61|85->61|86->62|91->67|91->67|91->67|92->68|108->84|108->84|108->84|109->85|109->85|109->85|110->86|110->86|110->86|111->87|112->88|113->89|113->89|113->89|149->125|149->125|149->125|153->129|153->129|153->129|154->130|154->130|154->130|154->130|155->131|163->139|163->139|163->139|168->144|168->144|168->144|172->148|172->148|172->148|176->152|176->152|176->152|180->156|180->156|180->156|184->160|184->160|184->160|188->164|188->164|188->164|192->168|192->168|192->168|196->172|196->172|196->172|200->176|200->176|200->176|214->190|214->190|214->190|218->194|218->194|218->194|219->195|220->196|220->196|220->196|221->197|222->198|222->198|222->198|230->206|230->206|230->206|232->208|232->208|232->208|233->209|233->209|233->209|234->210|234->210|234->210|239->215|239->215|239->215|240->216|241->217|241->217|241->217|242->218|243->219|243->219|243->219|286->262|286->262|287->263|288->264|288->264|293->269|293->269|294->270|295->271|295->271
                  -- GENERATED --
              */
          