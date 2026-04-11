// All articles, newest first
const articles = [
    {
        title: "Learning Diffusion #1: The Forward Process",
        summary: "Deriving the closed-form forward process for diffusion models from first principles — the noise schedule, sampling form of the Gaussian, the reparameterization trick, and how to jump to any noisy x_t directly from x_0.",
        category: "Machine Learning",
        date: "2026-04-11",
        readTime: "15 min read",
        url: "diffusion-forward-process.html"
    },
    {
        title: "Learning ML #4: Tiny Character-Level RNN on Shakespeare",
        summary: "Building a character-level language model from scratch using an RNN on the Tiny Shakespeare corpus. Covers embeddings, GRU, sliding-window datasets, CrossEntropyLoss flattening, perplexity, and greedy text generation.",
        category: "Machine Learning",
        date: "2026-04-06",
        readTime: "14 min read",
        url: "ml-char-rnn.html"
    },
    {
        title: "Learning ML #3: MNIST — From MLP to CNN",
        summary: "Classifying handwritten digits with torchvision, CrossEntropyLoss, and argmax accuracy. Built an MLP baseline first, then upgraded to a two-block convolutional network — and learned how spatial feature extraction actually works.",
        category: "Machine Learning",
        date: "2026-04-06",
        readTime: "12 min read",
        url: "ml-mnist.html"
    },
    {
        title: "Learning ML #2: Binary Classification on Circles",
        summary: "Training an MLP to separate two concentric rings of points. First encounter with BCEWithLogitsLoss, logits vs probabilities, overfitting in the wild, and computing accuracy properly.",
        category: "Machine Learning",
        date: "2026-04-06",
        readTime: "10 min read",
        url: "ml-circles.html"
    },
    {
        title: "Learning ML #1: Teaching a Neural Net to Fit a Parabola",
        summary: "The fundamentals of PyTorch from zero — tensor shapes, the training loop order, MSE loss, and a deep dive into SGD vs Adam vs AdamW. Built entirely from scratch without copying tutorial code.",
        category: "Machine Learning",
        date: "2026-04-06",
        readTime: "10 min read",
        url: "ml-parabola.html"
    },
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
