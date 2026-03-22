/* Chat sessions with history and right-side panel */
(function () {
  var STORAGE_KEY = "chat_sessions";

  var historyList = document.getElementById("chatHistoryList");
  var askAnything = document.getElementById("askAnything");

  var chatPanel = document.getElementById("chatPanel");
  var chatPanelClose = document.getElementById("chatPanelClose");
  var chatPanelTitle = document.getElementById("chatPanelTitle");
  var chatMessages = document.getElementById("chatMessages");
  var chatSendForm = document.getElementById("chatSendForm");
  var chatMessageInput = document.getElementById("chatMessageInput");

  if (!historyList || !chatPanel) return;

  function readSessions() {
    try {
      var raw = localStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch (e) {
      return [];
    }
  }

  function writeSessions(items) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
    } catch (e) {}
  }

  function createSession() {
    var sessions = readSessions();
    var id = "chat-" + Date.now();
    var session = { id: id, title: "New chat", messages: [] };
    sessions.unshift(session);
    writeSessions(sessions);
    return id;
  }

  function deleteSession(id) {
    var sessions = readSessions();
    var next = sessions.filter(function (s) { return s.id !== id; });
    writeSessions(next);
  }

  function getSession(id) {
    var sessions = readSessions();
    for (var i = 0; i < sessions.length; i++) {
      if (sessions[i].id === id) return sessions[i];
    }
    return null;
  }

  function updateSession(updated) {
    var sessions = readSessions();
    for (var i = 0; i < sessions.length; i++) {
      if (sessions[i].id === updated.id) {
        sessions[i] = updated;
        writeSessions(sessions);
        return;
      }
    }
  }

  function renderHistory() {
    var sessions = readSessions();
    // If no saved sessions but static lis exist, import them once
    if ((!sessions || sessions.length === 0) && historyList && historyList.children && historyList.children.length > 0) {
      var imported = [];
      for (var i = 0; i < historyList.children.length; i++) {
        var t = (historyList.children[i].textContent || '').trim();
        if (t) {
          imported.push({ id: 'chat-' + Date.now() + '-' + i, title: t, messages: [] });
        }
      }
      if (imported.length) {
        writeSessions(imported);
        sessions = imported;
      }
    }
    historyList.innerHTML = "";
    sessions.forEach(function (s) {
      var li = document.createElement("li");

      li.textContent = s.title || "New chat";
      li.title = s.title || "New chat";
      li.addEventListener("click", function () { openPanelWithSession(s.id); });

      historyList.appendChild(li);
    });
  }

  function openPanelWithSession(id) {
    var session = getSession(id);
    if (!session) return;
    chatPanel.setAttribute("data-chat-id", id);
    chatPanelTitle.textContent = session.title || "New chat";
    renderMessages(session.messages || []);
    // mark active item
    var lis = historyList.querySelectorAll('li');
    for (var i = 0; i < lis.length; i++) lis[i].classList.remove('active');
    for (var j = 0; j < lis.length; j++) {
      if ((lis[j].textContent || '').trim() === (session.title || 'New chat')) {
        lis[j].classList.add('active');
        break;
      }
    }
    chatPanel.classList.add("open");
    chatPanel.setAttribute("aria-hidden", "false");
    setTimeout(function(){ chatMessageInput && chatMessageInput.focus(); }, 50);
  }

  function closePanel() {
    chatPanel.classList.remove("open");
    chatPanel.setAttribute("aria-hidden", "true");
    chatPanel.removeAttribute("data-chat-id");
    chatMessages.innerHTML = "";
  }

  function renderMessages(messages) {
    chatMessages.innerHTML = "";
    messages.forEach(function (m) {
      var div = document.createElement("div");
      div.className = "chat-bubble" + (m.role === "user" ? " me" : "");
      div.textContent = m.text;
      chatMessages.appendChild(div);
    });
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function sendMessage(text) {
    var id = chatPanel.getAttribute("data-chat-id");
    if (!id) {
      id = createSession();
      openPanelWithSession(id);
    }
    var session = getSession(id);
    if (!session) return;
    session.messages.push({ role: "user", text: text });
    if (!session.title || session.title === "New chat") {
      session.title = text.slice(0, 40);
    }
    updateSession(session);
    chatPanelTitle.textContent = session.title;
    renderMessages(session.messages);
    renderHistory();
  }

  // Wire Ask Anything to start a new chat and open panel
  if (askAnything) {
    askAnything.addEventListener("click", function (e) {
      e.preventDefault();
      // If on chat.html, redirect the hero to focus
      try {
        var hero = document.getElementById('chatHeroText');
        if (hero) { hero.focus(); return; }
      } catch (e_) {}
      var id = createSession();
      openPanelWithSession(id);
    });
  }

  if (chatPanelClose) {
    chatPanelClose.addEventListener("click", function () { closePanel(); });
  }

  if (chatSendForm && chatMessageInput) {
    chatSendForm.addEventListener("submit", function (e) {
      e.preventDefault();
      var text = (chatMessageInput.value || "").trim();
      if (!text) return;
      sendMessage(text);
      chatMessageInput.value = "";
    });
  }

  renderHistory();

  // Optional: wire the hero form to create/open a chat
  try {
    var heroForm = document.getElementById('chatHeroForm');
    var heroInput = document.getElementById('chatHeroText');
    if (heroForm && heroInput) {
      heroForm.addEventListener('submit', function(ev) {
        ev.preventDefault();
        var text = (heroInput.value || '').trim();
        if (!text) return;
        var id = createSession();
        var session = getSession(id);
        session.title = text.slice(0, 60);
        updateSession(session);
        renderHistory();
        heroInput.value = '';
      });
    }
  } catch (e) {}
})();

