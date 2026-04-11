// All articles, newest first
const articles = [
    {
        title: "Diffusion #1: The Math Of The Gaussian Forward Pass",
        summary: "Deriving the closed-form forward process for diffusion models from first principles — the noise schedule, sampling form of the Gaussian, the reparameterization trick, and how to jump to any noisy x_t directly from x_0.",
        category: "Machine Learning",
        date: "2026-04-11",
        readTime: "15 min read",
        url: "diffusion-forward-process.html"
    },
    {
        title: "Real-Time Market Simulation with ESP32 Microcontrollers",
        summary: "Exploring my experimental market simulator that uses ESP32 trader nodes to study market microstructure and algorithmic trading mechanics with a TCP/UDP-based order book engine.",
        category: "Embedded",
        date: "2025-08-01",
        readTime: "12 min read",
        url: "market-simulation.html"
    },
    {
        title: "High-Reliability C++ Firmware for Distributed IoT Fleets",
        summary: "Lessons learned from engineering low-latency, deterministic firmware for real-time IoT devices — covering multi-threaded design, MQTT integration, and hard-won reliability principles.",
        category: "Embedded",
        date: "2026-01-06",
        readTime: "18 min read",
        url: "cpp-firmware.html"
    }
];

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
}

function renderPostList(containerId, limit) {
    const list = document.getElementById(containerId);
    if (!list) return;

    const sorted = [...articles].sort((a, b) => new Date(b.date) - new Date(a.date));
    const items = limit ? sorted.slice(0, limit) : sorted;

    list.innerHTML = items.map(article => `
        <li class="post-list-item">
            <div class="post-list-row">
                <span class="post-list-date">${formatDate(article.date)}</span>
                <a href="${article.url}" class="post-list-title">${article.title}</a>
            </div>
            <div class="post-list-summary">${article.summary}</div>
            <div class="post-list-meta">
                <span class="post-category">${article.category}</span>
                <span class="post-readtime">${article.readTime}</span>
            </div>
        </li>
    `).join('');
}

document.addEventListener('DOMContentLoaded', () => {
    renderPostList('articlesList');    // full list on articles.html
    renderPostList('recentPosts', 4); // preview on index.html
});
