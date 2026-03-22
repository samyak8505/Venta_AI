(function(){
  function updateFooterOffset(){
    var footer = document.querySelector('.app-footer');
    var sidebar = document.getElementById('sidebar');
    if(!footer || !sidebar) return;

    var rect = sidebar.getBoundingClientRect();
    var isCollapsed = sidebar.classList.contains('collapsed');
    var isToggled = sidebar.classList.contains('toggled');

    // If sidebar is toggled (off-canvas on mobile), don't offset footer
    var left = 0;
    if(!isToggled){
      // When collapsed, sidebar still has some width; use actual rendered width
      left = Math.max(0, rect.right);
    }

    // When sidebar is at x=0, rect.right equals its width
    // Cap left to a reasonable value (avoid full viewport on weird cases)
    var maxLeft = Math.min(left, window.innerWidth * 0.6);
    footer.style.left = maxLeft + 'px';
    footer.style.width = 'calc(100% - ' + maxLeft + 'px)';
  }

  ['resize','orientationchange'].forEach(function(evt){
    window.addEventListener(evt, updateFooterOffset);
  });

  // Hook into existing sidebar toggles if present
  try{
    var collapse = document.getElementById('btn-collapse');
    var toggle = document.getElementById('btn-toggle');
    var overlay = document.getElementById('overlay');
    if(collapse){ collapse.addEventListener('click', function(){ setTimeout(updateFooterOffset, 350); }); }
    if(toggle){ toggle.addEventListener('click', function(){ setTimeout(updateFooterOffset, 350); }); }
    if(overlay){ overlay.addEventListener('click', function(){ setTimeout(updateFooterOffset, 350); }); }
  }catch(e){}

  document.addEventListener('DOMContentLoaded', updateFooterOffset);
  // Also run a bit later to catch layout shifts
  setTimeout(updateFooterOffset, 500);
})();


