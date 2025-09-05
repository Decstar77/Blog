// Sample articles data
const articles = [
    {
        id: 1,
        title: "Real-Time Market Simulation with ESP32 Microcontrollers",
        excerpt: "Exploring my experimental market simulator project that uses ESP32 trader nodes to study market microstructure and algorithmic trading mechanics with TCP/UDP-based order book engine.",
        category: "projects",
        date: "2025-08-01",
        readTime: "12 min read",
        author: "Declan Porter"
    },
    {
        id: 2,
        title: "High-Reliability C++ Firmware for Distributed IoT Fleets",
        excerpt: "Lessons learned from engineering low-latency, deterministic firmware for real-time IoT devices with multi-threaded design and MQTT protocol integration.",
        category: "embedded",
        date: "2025-04-06",
        readTime: "18 min read",
        author: "Declan Porter"
    }
];

// DOM Elements
const hamburger = document.querySelector('.hamburger');
const navMenu = document.querySelector('.nav-menu');
const navLinks = document.querySelectorAll('.nav-link');
const articlesGrid = document.getElementById('articlesGrid');
const filterBtns = document.querySelectorAll('.filter-btn');

// Mobile Navigation
hamburger.addEventListener('click', () => {
    hamburger.classList.toggle('active');
    navMenu.classList.toggle('active');
});

// Close mobile menu when clicking on a link
navLinks.forEach(link => {
    link.addEventListener('click', () => {
        hamburger.classList.remove('active');
        navMenu.classList.remove('active');
    });
});

// Smooth scrolling for navigation links
navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const targetId = link.getAttribute('href');
        const targetSection = document.querySelector(targetId);
        
        if (targetSection) {
            targetSection.scrollIntoView({
                behavior: 'smooth'
            });
        }
        
        // Update active nav link
        navLinks.forEach(l => l.classList.remove('active'));
        link.classList.add('active');
    });
});

// Article filtering
let currentFilter = 'all';

filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const category = btn.getAttribute('data-category');
        
        // Update active filter button
        filterBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // Filter articles
        currentFilter = category;
        displayArticles();
    });
});

// Display articles based on current filter
function displayArticles() {
    const filteredArticles = currentFilter === 'all' 
        ? articles 
        : articles.filter(article => article.category === currentFilter);
    
    articlesGrid.innerHTML = '';
    
    filteredArticles.forEach(article => {
        const articleCard = createArticleCard(article);
        articlesGrid.appendChild(articleCard);
    });
}

// Create article card element
function createArticleCard(article) {
    const card = document.createElement('div');
    card.className = 'article-card';
    
    const categoryClass = getCategoryClass(article.category);
    
    card.innerHTML = `
        <span class="article-category ${categoryClass}">${article.category}</span>
        <h3 class="article-title">${article.title}</h3>
        <p class="article-excerpt">${article.excerpt}</p>
        <div class="article-meta">
            <div class="article-date">
                <i class="far fa-calendar"></i>
                <span>${formatDate(article.date)}</span>
            </div>
            <div class="read-time">
                <i class="far fa-clock"></i>
                <span>${article.readTime}</span>
            </div>
        </div>
    `;
    
    // Add click event to article card
    card.addEventListener('click', () => {
        // Check if this article has a dedicated page
        if (article.id === 1) {
            window.location.href = 'market-simulation.html';
        } else if (article.id === 2) {
            window.location.href = 'cpp-firmware.html';
        } else {
            showArticleModal(article);
        }
    });
    
    return card;
}

// Get category styling class
function getCategoryClass(category) {
    const categoryClasses = {
        'embedded': 'embedded-category',
        'iot': 'iot-category',
        'projects': 'projects-category',
        'tutorials': 'tutorials-category'
    };
    return categoryClasses[category] || '';
}

// Format date
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

// Show notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    // Style the notification
    notification.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        background: ${type === 'success' ? '#00b894' : '#ff6b6b'};
        color: ${type === 'success' ? '#000' : '#fff'};
        padding: 1rem 1.5rem;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        z-index: 10000;
        transform: translateX(100%);
        transition: transform 0.3s ease;
    `;
    
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 100);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// Article modal (simplified version)
function showArticleModal(article) {
    const modal = document.createElement('div');
    modal.className = 'article-modal';
    
    modal.innerHTML = `
        <div class="modal-overlay">
            <div class="modal-content">
                <div class="modal-header">
                    <h2>${article.title}</h2>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="article-meta">
                        <span>By ${article.author}</span>
                        <span>${formatDate(article.date)}</span>
                        <span>${article.readTime}</span>
                    </div>
                    <p>${article.excerpt}</p>
                    <p>This is a preview of the article. The full content would be displayed here with detailed analysis, code examples, and comprehensive insights into ${article.category} topics.</p>
                </div>
            </div>
        </div>
    `;
    
    // Style the modal
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: 10000;
        display: flex;
        align-items: center;
        justify-content: center;
    `;
    
    const overlay = modal.querySelector('.modal-overlay');
    overlay.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.8);
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px;
    `;
    
    const content = modal.querySelector('.modal-content');
    content.style.cssText = `
        background: #1a1a1a;
        border: 1px solid rgba(0, 184, 148, 0.3);
        border-radius: 12px;
        max-width: 600px;
        width: 100%;
        max-height: 80vh;
        overflow-y: auto;
        position: relative;
    `;
    
    const header = modal.querySelector('.modal-header');
    header.style.cssText = `
        padding: 2rem 2rem 1rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
    `;
    
    header.querySelector('h2').style.cssText = `
        color: #ffffff;
        margin: 0;
        font-size: 1.5rem;
        line-height: 1.4;
    `;
    
    const closeBtn = modal.querySelector('.modal-close');
    closeBtn.style.cssText = `
        background: none;
        border: none;
        color: #888;
        font-size: 2rem;
        cursor: pointer;
        padding: 0;
        line-height: 1;
    `;
    
    const body = modal.querySelector('.modal-body');
    body.style.cssText = `
        padding: 1rem 2rem 2rem;
    `;
    
    body.querySelector('.article-meta').style.cssText = `
        display: flex;
        gap: 1rem;
        color: #888;
        font-size: 0.9rem;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    `;
    
    body.querySelectorAll('p').forEach(p => {
        p.style.cssText = `
            color: #b0b0b0;
            line-height: 1.6;
            margin-bottom: 1rem;
        `;
    });
    
    document.body.appendChild(modal);
    
    // Close modal functionality
    const closeModal = () => {
        document.body.removeChild(modal);
    };
    
    closeBtn.addEventListener('click', closeModal);
    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) closeModal();
    });
    
    // Close on escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeModal();
    });
}

// Intersection Observer for scroll animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe elements for animation
document.addEventListener('DOMContentLoaded', () => {
    const animatedElements = document.querySelectorAll('.article-card, .expertise-item, .stat');
    
    animatedElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
    
    // Initialize articles display
    displayArticles();
});

// Simple chart animation for hero section
function createHeroChart() {
    const canvas = document.getElementById('heroChart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    canvas.width = 400;
    canvas.height = 200;
    
    // Create a simple animated line chart
    const data = [20, 35, 25, 45, 30, 55, 40, 65, 50, 75, 60, 85];
    const step = canvas.width / (data.length - 1);
    
    ctx.strokeStyle = '#00b894';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    data.forEach((value, index) => {
        const x = index * step;
        const y = canvas.height - (value / 100) * canvas.height;
        
        if (index === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    
    ctx.stroke();
    
    // Add some dots
    ctx.fillStyle = '#00b894';
    data.forEach((value, index) => {
        const x = index * step;
        const y = canvas.height - (value / 100) * canvas.height;
        
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fill();
    });
}

// Initialize chart when page loads
document.addEventListener('DOMContentLoaded', createHeroChart);

// Add some interactive stats animation
function animateStats() {
    const stats = document.querySelectorAll('.stat-number');
    
    stats.forEach(stat => {
        const target = parseInt(stat.textContent);
        let current = 0;
        const increment = target / 50;
        
        const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                current = target;
                clearInterval(timer);
            }
            stat.textContent = Math.floor(current) + (stat.textContent.includes('+') ? '+' : '') + (stat.textContent.includes('%') ? '%' : '');
        }, 50);
    });
}

// Trigger stats animation when hero section is visible
const heroObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            animateStats();
            heroObserver.unobserve(entry.target);
        }
    });
}, { threshold: 0.5 });

document.addEventListener('DOMContentLoaded', () => {
    const heroSection = document.querySelector('.hero');
    if (heroSection) {
        heroObserver.observe(heroSection);
    }
}); 