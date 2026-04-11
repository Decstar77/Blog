// ── Background: procedural triangle mesh ────────────────────
(function () {
    const canvas = document.createElement('canvas');
    Object.assign(canvas.style, {
        position: 'fixed', top: '0', left: '0',
        width: '100%', height: '100%',
        zIndex: '-1', pointerEvents: 'none', display: 'block'
    });

    // Insert before any body content so it's immediately behind everything
    if (document.body) {
        document.body.insertBefore(canvas, document.body.firstChild);
    } else {
        document.addEventListener('DOMContentLoaded', () =>
            document.body.insertBefore(canvas, document.body.firstChild)
        );
    }

    const ctx = canvas.getContext('2d');
    let tris = null, cachedW = 0, cachedH = 0;

    // Seeded LCG — same layout every load
    function makeLCG(seed) {
        let s = seed >>> 0;
        return () => {
            s = (Math.imul(1664525, s) + 1013904223) >>> 0;
            return s / 4294967296;
        };
    }

    function buildTriangles(w, h) {
        const rand = makeLCG(0xC0FFEE42);
        const result = [];
        const count = 65;
        for (let i = 0; i < count; i++) {
            const cx  = rand() * w;
            const cy  = rand() * h;
            const sz  = 16 + rand() * 72;
            const rot = rand() * Math.PI * 2;
            const pts = [0, 1, 2].map(j => {
                const a = rot + j * (Math.PI * 2 / 3) + (rand() - 0.5) * 0.85;
                const r = sz * (0.5 + rand() * 0.65);
                return [cx + Math.cos(a) * r, cy + Math.sin(a) * r];
            });
            const green = rand() < 0.18;   // ~1 in 6 triangles gets green accent
            const alpha = 0.05 + rand() * 0.18;
            result.push({ pts, green, alpha });
        }
        return result;
    }

    function draw() {
        const w = window.innerWidth;
        const h = window.innerHeight;
        canvas.width  = w;
        canvas.height = h;

        // Rebuild triangle layout only when viewport changes noticeably
        if (!tris || Math.abs(w - cachedW) > 40 || Math.abs(h - cachedH) > 40) {
            tris = buildTriangles(w, h);
            cachedW = w; cachedH = h;
        }

        // Dark background fill
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, 0, w, h);

        for (const { pts, green, alpha } of tris) {
            ctx.beginPath();
            ctx.moveTo(pts[0][0], pts[0][1]);
            ctx.lineTo(pts[1][0], pts[1][1]);
            ctx.lineTo(pts[2][0], pts[2][1]);
            ctx.closePath();

            if (green) {
                ctx.fillStyle   = `rgba(45, 164, 78, ${alpha * 0.28})`;
                ctx.fill();
                ctx.strokeStyle = `rgba(60, 200, 90, ${alpha + 0.18})`;
                ctx.lineWidth   = 1.2;
            } else {
                ctx.strokeStyle = `rgba(210, 210, 210, ${alpha})`;
                ctx.lineWidth   = 0.7;
            }
            ctx.stroke();
        }
    }

    let resizeTimer;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(draw, 120);
    });

    draw();
})();

// ── Navigation & smooth scroll ───────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    const hamburger = document.querySelector('.hamburger');
    const nav = document.querySelector('.site-nav');

    if (hamburger && nav) {
        hamburger.addEventListener('click', () => nav.classList.toggle('open'));
        nav.querySelectorAll('a').forEach(a =>
            a.addEventListener('click', () => nav.classList.remove('open'))
        );
    }

    document.querySelectorAll('a[href^="#"]').forEach(link => {
        link.addEventListener('click', e => {
            const target = document.querySelector(link.getAttribute('href'));
            if (target) {
                e.preventDefault();
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
});
