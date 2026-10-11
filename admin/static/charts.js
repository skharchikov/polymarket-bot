// Draws every [data-chart=equity] element from the JSON in its data-src <script>.
function drawEquity(el) {
  const src = document.getElementById(el.dataset.src);
  if (!src) return;
  const d = JSON.parse(src.textContent);
  if (!d.days.length) { el.textContent = "No resolved bets in range."; return; }
  const axis = { stroke: "#7d869a", grid: { stroke: "#222736" } };
  new uPlot({
    width: el.clientWidth || 600,
    height: 180,
    series: [
      {},
      { label: "trading", stroke: "#5b8cff", width: 2 },
      { label: "copy", stroke: "#f5b942", width: 2 },
    ],
    axes: [axis, axis],
  }, [d.days, d.trading, d.copy], el);
}
document.addEventListener("DOMContentLoaded", () =>
  document.querySelectorAll("[data-chart=equity]").forEach(drawEquity));
