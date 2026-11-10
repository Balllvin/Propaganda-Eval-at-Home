(function () {
  "use strict";
  var R = window.__report || {}, items = R.items || [], models = R.models || [];
  var $ = function (id) { return document.getElementById(id); };
  var el = function (tag, cls, text) { var n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };
  var pct = function (x) { return x == null ? "n/a" : (100 * x).toFixed(1) + "%"; };
  var fx = function (x) { return x == null ? "n/a" : Number(x).toFixed(2); };

  /* headline */
  var judges = {};
  models.forEach(function (m) {
    var s = m.summary, c = el("div", "card");
    c.appendChild(el("h3", null, m.name));
    c.appendChild(el("div", "muted", m.model + " · " + m.profile + " · " + m.runs + " run" + (m.runs > 1 ? "s" : "")));
    c.appendChild(el("div", "num", fx(s.fact_mean)));
    c.appendChild(el("div", "ci", "fact score, 95% " + fx(s.fact_mean_ci95[0]) + " to " + fx(s.fact_mean_ci95[1])));
    c.appendChild(el("div", "num", pct(s.any_propaganda)));
    c.appendChild(el("div", "ci", "any propaganda, 95% " + pct(s.any_propaganda_ci95[0]) + " to " + pct(s.any_propaganda_ci95[1])));
    c.appendChild(el("div", "muted", "graded " + s.graded + " of " + s.prompts + ", unanswered " + s.unanswered + ", refused " + s.refused));
    if (m.runs > 1) c.appendChild(el("div", "muted", "fact score by run: " + m.fact_by_run.map(fx).join(", ")));
    $("cards").appendChild(c);
    (s.judges || []).forEach(function (j) { judges[j] = 1; });
  });
  var js = Object.keys(judges);
  if (js.some(function (j) { return j.split("/").pop() !== "grok-4.7"; })) {
    $("warn").hidden = false;
    $("warn").textContent = "Judges in this report: " + js.join(", ") + ". Scores from a judge other than grok-4.7 are not comparable with the Nous leaderboard or with each other.";
  }

  /* group tables */
  function table(id, key) {
    var t = $(id), head = el("tr");
    head.appendChild(el("th", null, key === "by_type" ? "Question type" : "Topic"));
    head.appendChild(el("th", null, "n"));
    models.forEach(function (m) { head.appendChild(el("th", null, m.name + " fact")); head.appendChild(el("th", null, m.name + " propaganda")); });
    t.appendChild(head);
    var names = models.length ? Object.keys(models[0].summary[key]) : [];
    names.forEach(function (g) {
      var tr = el("tr");
      tr.appendChild(el("td", null, g));
      tr.appendChild(el("td", null, String(models[0].summary[key][g].n)));
      models.forEach(function (m) { var v = m.summary[key][g] || {}; tr.appendChild(el("td", null, fx(v.fact_mean))); tr.appendChild(el("td", null, pct(v.any_propaganda))); });
      t.appendChild(tr);
    });
  }
  table("bytype", "by_type");
  table("bycat", "by_category");

  /* filters */
  function options(sel, values) { values.forEach(function (v) { var o = el("option", null, v); o.value = v; sel.appendChild(o); }); }
  models.forEach(function (m, i) { var o = el("option", null, m.name); o.value = String(i); $("f-model").appendChild(o); });
  var types = {}, cats = {}, stats = {};
  items.forEach(function (it) { types[it.q] = 1; it.c.forEach(function (c) { cats[c] = 1; }); });
  models.forEach(function (m) { Object.keys(m.rows).forEach(function (k) { stats[m.rows[k].s] = 1; }); });
  options($("f-type"), Object.keys(types).sort());
  options($("f-cat"), Object.keys(cats).sort());
  options($("f-status"), Object.keys(stats).sort());

  var shown = 0, matches = [], PAGE = 50;
  function pre(label, text, parent) { parent.appendChild(el("h4", null, label)); parent.appendChild(el("pre", null, text || "(empty)")); }
  function render(it, m) {
    var row = m.rows[it.id], v = row.v, li = el("li"), d = el("details"), sum = el("summary");
    sum.appendChild(el("span", "tag", it.id));
    sum.appendChild(el("span", "tag", v ? "fact " + v.score : row.s));
    if (v) sum.appendChild(el("span", "tag p" + v.propaganda, "propaganda " + v.propaganda));
    sum.appendChild(el("span", null, it.p));
    d.appendChild(sum);
    var body = el("div", "body");
    body.appendChild(el("div", "muted", it.q + " · " + it.c.join(", ") + " · finish " + row.f + (row.t != null ? " · " + row.t + " tokens" : "")));
    if (it.u) { var a = el("a", null, "source"); a.href = it.u; a.rel = "noreferrer"; body.appendChild(a); }
    pre("Required facts", it.rf.map(function (f) { return "- " + f; }).join("\n"), body);
    var r = el("details"); r.appendChild(el("summary", null, "Reasoning (" + row.r.length + " characters)")); r.appendChild(el("pre", null, row.r || "(none)")); body.appendChild(r);
    pre("Answer", row.a, body);
    pre("Verdict", v ? JSON.stringify(v, null, 1) : "not graded (counts as fact 0, propaganda 0)", body);
    d.appendChild(body); li.appendChild(d); return li;
  }
  function apply() {
    var m = models[+$("f-model").value || 0]; if (!m) return;
    var t = $("f-type").value, c = $("f-cat").value, f = $("f-fact").value, p = $("f-prop").value, s = $("f-status").value, q = $("f-q").value.toLowerCase();
    matches = items.filter(function (it) {
      var row = m.rows[it.id], v = row.v;
      if (t && it.q !== t) return false;
      if (c && it.c.indexOf(c) < 0) return false;
      if (f === "u" && v) return false;
      if (f && f !== "u" && (!v || String(v.score) !== f)) return false;
      if (p && (!v || String(v.propaganda) !== p)) return false;
      if (s && row.s !== s) return false;
      if (q && (it.id + " " + it.p + " " + row.a).toLowerCase().indexOf(q) < 0) return false;
      return true;
    });
    $("list").textContent = ""; shown = 0; more();
    $("count").textContent = matches.length + " of " + items.length + " items for " + m.name;
  }
  function more() {
    var m = models[+$("f-model").value || 0];
    matches.slice(shown, shown + PAGE).forEach(function (it) { $("list").appendChild(render(it, m)); });
    shown += PAGE; $("more").hidden = shown >= matches.length;
  }
  $("filters").addEventListener("input", apply);
  $("filters").addEventListener("submit", function (e) { e.preventDefault(); });
  $("more").addEventListener("click", more);
  apply();
})();
