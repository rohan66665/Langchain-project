const chatBox = document.getElementById("chat");
const q = document.getElementById("q");
const askBtn = document.getElementById("ask");
const clearBtn = document.getElementById("clear");

function addMsg(role, text) {
  const div = document.createElement("div");
  div.className = "msg " + (role === "user" ? "user" : "bot");
  div.textContent = (role === "user" ? "You: " : "Bot: ") + text;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function addSources(sources) {
  if (!sources || !sources.length) return;
  const ul = document.createElement("div");
  ul.className = "src";
  ul.textContent = "Sources: " + sources.map(s => s.source).join(", ");
  chatBox.appendChild(ul);
}

async function sendMessage(reset=false) {
  const msg = reset ? "" : (q.value || "").trim();
  if (!reset && !msg) return;

  if (!reset) addMsg("user", msg);
  q.value = "";
  const body = reset ? { reset: true, message: "" } : { message: msg };

  const res = await fetch("/chat", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(body)
  });
  const data = await res.json();
  addMsg("bot", data.answer || "(no answer)");
  addSources(data.sources);
}

askBtn.onclick = () => sendMessage(false);
clearBtn.onclick = () => sendMessage(true);
q.addEventListener("keydown", (e) => { if (e.key === "Enter") sendMessage(false) });
