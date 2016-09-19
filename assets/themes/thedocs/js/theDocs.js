
$(function() {

  "use strict";

  // Back to top
  $('#scroll-up').on('click', function() {
    $('html, body').animate({scrollTop : 0}, 900);
    return false;
  });

  // Smooth scroll for ToC
  $('.toc a, .sidenav.nav a').click(function(){
    $('html, body').animate({scrollTop: $($.attr(this, 'href')).offset().top - 80}, 500);
    return false;
  });

  // Smoothscroll to anchor in page load
  var hash = location.hash.replace('#','');
  if (hash != '' && $("#"+hash).length > 0) {
    $('html, body').animate({scrollTop: $("#"+hash).offset().top-100}, 600);
  }


  // Full height body to make sure footer will place in bottom of the page
  if ($(window).height() > $('body').height()) {
    var min_height = $(window).height() - $('.site-header').height() - $('.site-footer').height() - 60;
    $('body > main').css('min-height', min_height);
  }

  // Set the height of sidebar if it's fixed  .height(sidenav_max_height)
  if ($('.sidenav.sticky').length > 0) {
    var sidenav_max_height = $(window).height() - $('.sidenav.sticky').position().top - 100;
    $('.sidenav.sticky').height(sidenav_max_height);
  }

  //
  // Top navbar
  //
  if ($('.site-header').hasClass('sticky') && !$('.site-header').hasClass('navbar-sm')) {
    var navbar_lg = false;
    if ($('.site-header').hasClass('navbar-lg')) {
      navbar_lg = true;
    }

    $(window).on('scroll', function() {
      var offset = $('.site-header').offset().top + $('.site-header').height();

      if ($(window).scrollTop() > offset) {
        if (navbar_lg) {
          $('.site-header').removeClass('navbar-lg');
        }
        $('.site-header').addClass('navbar-sm');

      }
      else {
        if (navbar_lg) {
          $('.site-header').addClass('navbar-lg');
        }
        $('.site-header').removeClass('navbar-sm');
      }
    });
  }

  // Manage transparent navbar
  if ($('.site-header').hasClass('navbar-transparent') && $('.site-header').hasClass('sticky')) {

    if ($('.site-header > .banner').length == 0) {
      $('.site-header').removeClass('navbar-transparent');
    }
    else {
      if ($('.site-header').hasClass('sticky')) {

        $(window).on('scroll', function() {
          var offset = $('.site-header .navbar').height();
          if ($(window).scrollTop() > offset) {
            $('.site-header').removeClass('navbar-transparent')
          }
          else {
            $('.site-header').addClass('navbar-transparent')
          }
        });

      }
    }

  }

  // Margin top for sticky navbar without banner
  if ($('.site-header').hasClass('sticky') && $('.site-header > .banner').length == 0) {
    $('.site-header').css('padding-top', $('.site-header > .navbar').height() + 30);
  }

  // Add .force-middle if navbar-brand contains image
  if ('.navbar-brand > img') {
    $('.navbar-brand').prepend('<span class="force-middle"></span>');
  }


  //
  // Sidebar
  //

  // Offcanvas
  $('[data-toggle="offcanvas"]').on('click', function () {
    //$('.main-content').css('height', $('.sidenav').height()+100 + 'px');
    $('body').toggleClass('open-sidebar');
    if ($('body').hasClass('open-sidebar')) {
      $('html').css('overflow', 'hidden');
      //$('.main-content').css('height', $('.sidenav').height()+100 + 'px');
      $('.site-header .jumbotron').slideUp(50);
    }
    else {
      $('html').css('overflow', 'visible');
      //$('.main-content').css('height', 'auto');
      $('.site-header .jumbotron').slideDown(900);
    }
  });

  // Dropdown
  $('.sidenav.dropable > li > a').on('click', function(e){

    if ( 0 < $(this).next("ul").length ) {
      e.preventDefault();
    }

    if ( 0 == $(this).next("ul").length ) {
      return;
    }

	if ( $(this).hasClass('open') ) {
	  $(this).removeClass('open').next("ul").slideUp(300);
	  return;
	}

    $(this).parents(".sidenav").find("> li > a").removeClass('open');
    $(this).parents(".sidenav").find("ul").not(":hidden").slideUp(300);
    $(this).addClass('open').next("ul").slideDown(300);
  });

  $('.sidenav.dropable > li > a.active').addClass('open');
  $('.sidenav.dropable > li > ul').prev('a').addClass('has-child');

  if ($(window).width() < 768) {
    $('.sidebar-boxed').removeClass('sidebar-dark');
  }

  // Sticky behaviour
  if ($('.sidenav').hasClass('sticky')) {
    $(window).on('scroll', function() {
      var $sidenav = $('.sidenav'),
        offset   = $('.sidebar').offset();

      if ($(window).scrollTop() > offset.top) {
        $sidenav.css({ position: 'fixed', top: '120px', width: '215px' });
      } else {
        $sidenav.css('position', 'static');
      }
    });
  }

  // Auto link creator for headings
  $('h1[id], h2[id], h3[id], h4[id], h5[id], h6[id]').each(function(index, value) {
    var link = '<a href="#'+ $(this).attr("id") +'">'+ $(this).html() +'</a>';
    $(this).html(link);
  });

  //
  // FAQ Component
  //

  // Case insensitive contains selector
  jQuery.expr[':'].icontains = function(a, i, m) {
    return jQuery(a).text().toUpperCase().indexOf(m[3].toUpperCase()) >= 0;
  };

  // Search
  $('.faq-search').on('keyup', function(e) {
    var s = $(this).val().trim(),
      result = $(this).parent().find("li");
    if (s === '') {
      result.show();
      return true;
    }
    result.not(':icontains(' + s + ')').hide();
    result.filter(':icontains(' + s + ')').show();
  });

  $('.faq li > h6').on('click', function() {
    $(this).toggleClass('open').next('div').slideToggle(300);
  });

  //Taking care of video
  if ($.fn.mediaelementplayer) {
    $('video').mediaelementplayer();
  }

  if ($.fn.fitVids) {
    $('.video').fitVids();
  }

  //
  // File Tree
  //
  $('.file-tree li.is-file').on('click', function(e){
    e.stopPropagation();
  });

  $('.file-tree li.is-folder').on('click', function(e){
    $(this).find('ul:first').slideToggle(400, function(){
      $(this).parent('li').toggleClass('open');
    });
    e.stopPropagation();
  });


  //Equal height for grid view
  $('.grid-view > li, .categorized-view > li, .promo.small-icon').matchHeight();

  //
  // Code viewers
  //

  // Copy to clipboard
  // It doesn't support Safari yet, and also has some minor bugs
  $('pre').each(function(index, value) {
    $(this).prepend('<a class="btn btn-sm btn-purple clipboard-copy" data-original-title="Copied!">Copy</a>');
  });

  // Code snippet
  $('pre').each(function(index, value) {
    if ($(this).parents('.code-window').length || $(this).parents('.code-taps').length) {
      return;
    }
    var title = "";
    if ($(this).children("code").attr('class')) {
      title = $(this).children("code").attr('class');
      title = title.replace("language-", "");
      title = title.toLowerCase();
      if (title == "markup") {
        title = "html";
      }
    }
    var span = '<span class="language-name">'+ title +'</span>';
    $(this).prepend(span);
  });

  $('pre .language-name').parent().on('scroll', function(){
    $(this).find('.language-name').css('transform', 'translate('+ $(this).scrollLeft() +'px, '+ $(this).scrollTop() +'px)');
  });

  // Code window
  $('.code-window').each(function(index, value){
    var topbar = '<div class="window-bar"><div class="circles">';
    topbar += '<span class="circle circle-red"></span> <span class="circle circle-yellow"></span> <span class="circle circle-green"></span>';
    if ($(this).attr('data-title')) {
      topbar += '<span class="window-title">'+ $(this).data('title') +'</span>';
    }
    topbar += '</div>';//.circles

    //Languages
    if ($(this).children().length > 1) {
      topbar += '<div class="languages"><div class="btn-group" data-toggle="buttons">';

      $(this).children().each(function(index, value){
        var active='', check='', title='';
        if (index == 0) {
          active = ' active';
          check = ' checked';
        }
        if ($(this).children("code").attr('class')) {
          title = $(this).children("code").attr('class');
          title = title.replace("language-", "");
          title = title.toLowerCase();
          if (title == "markup") {
            title = "html";
          }
        }
        else if ($(this).hasClass('code-preview')) {
          title = 'Example';
        }
        topbar += '<label class="btn'+ active +'"><input type="radio" autocomplete="off"'+ check +'>'+ title +'</label>';
      });

      topbar += '</div></div>';
    }

    topbar += '</div>';//.window-bar

    $(this).children(':not(:first)').hide(0);
    $(this).children().wrapAll('<div class="window-content"></div>');
    $(this).prepend(topbar);

    //Event handler, change tab
    var window_content = $(this).children('.window-content');
    $(this).find(".btn-group .btn").on('click', function() {
      var i = $(this).index();
      window_content.children(":visible").fadeOut(200, function() {
        window_content.children(":not(.prism-show-language):eq("+ i +")").fadeIn(200);
      });
    });
  });

  // Code tabs
  $('.code-tabs').each(function(index, value){
    var topbar = '';

    //Languages
    if ($(this).children().length > 1) {
      topbar += '<div class="languages"><div class="btn-group" data-toggle="buttons">';

      $(this).children().each(function(index, value){
        var active='', check='', title='';
        if (index == 0) {
          active = ' active';
          check = ' checked';
        }
        if ($(this).children("code").attr('class')) {
          title = $(this).children("code").attr('class');
          title = title.replace("language-", "");
          title = title.toLowerCase();
          if (title == "markup") {
            title = "html";
          }
        }
        else if ($(this).hasClass('code-preview')) {
          title = 'Example';
        }
        topbar += '<label class="btn'+ active +'"><input type="radio" autocomplete="off"'+ check +'>'+ title +'</label>';
      });

      topbar += '</div></div>';
    }

    $(this).children(':not(:first)').hide(0);
    $(this).children().wrapAll('<div class="window-content"></div>');
    $(this).prepend(topbar);

    //Event handler, change tab
    var window_content = $(this).children('.window-content');
    $(this).find(".btn-group .btn").on('click', function() {
      var i = $(this).index();
      window_content.children(":visible").fadeOut(200, function() {
        window_content.children(":not(.prism-show-language):eq("+ i +")").fadeIn(200);
      });
    });
  });

  // Trim code blocks
  $('pre code').each(function(){
    $(this).html($.trim($(this).html()));
  });


  // Copy to clipboard
  $('.code-preview .clipboard-copy').remove();
  $('.clipboard-copy').tooltip({placement: 'bottom', trigger: 'manual'});
  // Move copy button when the content is scrolling
  $('.clipboard-copy').parent().on('scroll', function(){
    $(this).find('.clipboard-copy').css('transform', 'translate('+ $(this).scrollLeft() +'px, '+ $(this).scrollTop() +'px)');
  });

  if ($('.clipboard-copy').length > 0) {

    var clipboardSnippets = new Clipboard('.clipboard-copy', {
      target: function(trigger) {
        return trigger.nextElementSibling;
      }
    });

    clipboardSnippets.on('success', function(e) {
      e.clearSelection();
      $(e.trigger).tooltip('show');
      setTimeout(function(el){ $(el.trigger).tooltip('hide'); }, 1000, e);
    });
  }


});
