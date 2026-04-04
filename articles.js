// All articles, newest first
const articles = [
    {
        title: "Real-Time Market Simulation with ESP32 Microcontrollers",
        summary: "Exploring my experimental market simulator that uses ESP32 trader nodes to study market microstructure and algorithmic trading mechanics with a TCP/UDP-based order book engine.",
        category: "Projects",
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
    return date.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
}

function renderArticles() {
    const list = document.getElementById('articlesList');
    if (!list) return;

    // Sort newest first
    const sorted = [...articles].sort((a, b) => new Date(b.date) - new Date(a.date));

    list.innerHTML = sorted.map(article => `
        <a href="${article.url}" class="article-list-item">
            <div class="article-list-meta">
                <span class="article-list-date">${formatDate(article.date)}</span>
                <span class="article-list-category">${article.category}</span>
            </div>
            <div class="article-list-title">${article.title}</div>
            <div class="article-list-summary">${article.summary}</div>
            <div class="article-list-readtime">
                <i class="far fa-clock"></i>
                ${article.readTime}
            </div>
        </a>
    `).join('');
}

// Mobile nav toggle
document.querySelector('.hamburger').addEventListener('click', () => {
    document.querySelector('.hamburger').classList.toggle('active');
    document.querySelector('.nav-menu').classList.toggle('active');
});

document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', () => {
        document.querySelector('.hamburger').classList.remove('active');
        document.querySelector('.nav-menu').classList.remove('active');
    });
});

document.addEventListener('DOMContentLoaded', renderArticles);
