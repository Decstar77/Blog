# Declan Porter - Embedded Systems Blog

A personal blog showcasing embedded systems projects, IoT development, and creative technology work. Built with vanilla HTML, CSS, and JavaScript for simplicity and performance.

## Features

### 🎨 Design
- **Modern Dark Theme**: Professional dark color scheme with cyan accents
- **Responsive Design**: Fully responsive layout that works on all devices
- **Clean Typography**: Uses Inter font family for excellent readability
- **Smooth Animations**: Subtle animations and transitions for enhanced UX

### 📱 User Experience
- **Fixed Navigation**: Sticky header with smooth scrolling navigation
- **Mobile Menu**: Hamburger menu for mobile devices
- **Article Filtering**: Filter articles by category (Trading, Risk Management, Technology, Research)
- **Interactive Cards**: Hover effects and click interactions on article cards
- **Newsletter Subscription**: Email subscription form with notifications

### 📊 Content Features
- **Sample Articles**: 8 realistic articles covering embedded systems, IoT, and personal projects
- **Category System**: Organized content by embedded, IoT, projects, and tutorials
- **Article Modals**: Click to view detailed article previews
- **Author Information**: Each article includes author and read time
- **Date Formatting**: Properly formatted publication dates

### 🚀 Technical Features
- **Vanilla JavaScript**: No frameworks or dependencies required
- **Canvas Animation**: Simple chart animation in hero section
- **Intersection Observer**: Scroll-triggered animations
- **Local Storage**: Newsletter subscription handling
- **Cross-browser Compatible**: Works on all modern browsers

## File Structure

```
Blog/
├── index.html          # Main HTML file
├── styles.css          # All CSS styles
├── script.js           # JavaScript functionality
└── README.md           # This file
```

## Getting Started

### Prerequisites
- A modern web browser (Chrome, Firefox, Safari, Edge)
- No additional software or dependencies required

### Installation
1. Clone or download this repository
2. Open `index.html` in your web browser
3. That's it! The blog is ready to use

### Local Development
For local development, you can use any simple HTTP server:

```bash
# Using Python 3
python -m http.server 8000

# Using Node.js (if you have http-server installed)
npx http-server

# Using PHP
php -S localhost:8000
```

Then visit `http://localhost:8000` in your browser.

## Customization

### Adding New Articles
Edit the `articles` array in `script.js`:

```javascript
const articles = [
    {
        id: 9,
        title: "Your New Article Title",
        excerpt: "Article description...",
        category: "embedded", // or "iot", "projects", "tutorials"
        date: "2024-01-20",
        readTime: "10 min read",
        author: "Declan Porter"
    },
    // ... more articles
];
```

### Changing Colors
Modify the CSS variables in `styles.css`:

```css
/* Primary accent color */
--accent-color: #00d4ff;

/* Background colors */
--bg-primary: #0a0a0a;
--bg-secondary: #1a1a1a;
```

### Adding New Categories
1. Add the category to the filter buttons in `index.html`
2. Update the `getCategoryClass()` function in `script.js`
3. Add corresponding CSS styles if needed

### Personalizing Content
The blog is now personalized for Declan Porter's background in:
- C++ embedded programming at Eranest
- IoT device development (Aqua Scanner, Aqua Meter)
- ESP32 microcontroller projects
- Open-source contributions
- Creative projects (Blender addons, market simulation)

## Content Categories

### Embedded Systems
- High-performance C++ firmware
- Real-time systems design
- Multi-threaded applications
- ESP32 development

### IoT Development
- MQTT protocol implementation
- Distributed IoT fleets
- Edge computing
- Water conservation systems

### Projects
- Market simulation with ESP32
- Blender Python addons
- Open-source contributions
- Experimental systems

### Tutorials
- ESP32 getting started guides
- C++ embedded programming
- IoT best practices
- Development tools and setup

## Browser Support

- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## Performance

- **Lightweight**: No external dependencies
- **Fast Loading**: Optimized CSS and JavaScript
- **SEO Friendly**: Semantic HTML structure
- **Accessible**: Proper ARIA labels and keyboard navigation

## License

This project is open source and available under the [MIT License](LICENSE).

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Contact

For questions or suggestions, please open an issue on GitHub.

---

Built with ❤️ for the embedded systems and IoT community. 