// All articles, newest first
const articles = [
    {
        title: "C++26 Reflection: A JSON Serializer in Forty Lines",
        summary: "Building a struct-to-JSON writer with the ^^ reflection operator, splicers and template for, and what the four rewrites taught me.",
        category: "C++",
        readTime: "14 min read",
        url: "cpp26-reflection-json.html"
    },
    {
        title: "The Math Behind Variational Autoencoders",
        summary: "From the intractable evidence integral to the ELBO, deriving the variational lower bound, the reparameterisation trick, and the closed-form KL.",
        category: "Machine Learning",
        readTime: "25 min read",
        url: "vae-math.html"
    },
    {
        title: "Diffusion #2: The Math Of The Gaussian Backward Pass",
        summary: "Deriving the reverse diffusion posterior from Bayes' theorem.",
        category: "Machine Learning",
        readTime: "20 min read",
        url: "diffusion-backward-pass.html"
    },
    {
        title: "Diffusion #1: The Math Of The Gaussian Forward Pass",
        summary: "Deriving the closed-form forward process for diffusion models from first principles. ",
        category: "Machine Learning",
        readTime: "15 min read",
        url: "diffusion-forward-process.html"
    },
    // Don't show these anymore 
    // {
    //     title: "Real-Time Market Simulation with ESP32 Microcontrollers",
    //     summary: "Exploring my experimental market simulator that uses ESP32 trader nodes to study market microstructure and algorithmic trading mechanics with a TCP/UDP-based order book engine.",
    //     category: "Embedded",
    //     readTime: "12 min read",
    //     url: "market-simulation.html"
    // },
    // {
    //     title: "High-Reliability C++ Firmware for Distributed IoT Fleets",
    //     summary: "Lessons learned from engineering low-latency, deterministic firmware for real-time IoT devices — covering multi-threaded design, MQTT integration, and hard-won reliability principles.",
    //     category: "Embedded",
    //     readTime: "18 min read",
    //     url: "cpp-firmware.html"
    // }
];

function renderPostList(containerId, limit) {
    const list = document.getElementById(containerId);
    if (!list) return;

    const items = limit ? articles.slice(0, limit) : articles;

    list.innerHTML = items.map(article => `
        <li class="post-list-item">
            <div class="post-list-row">
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
