using System.Numerics;
using Raylib_cs;

namespace arcade {
    // Immediate-mode widgets drawn in the Style palette. Call them every frame between Begin and End.
    // Widgets are identified by call order, so that order has to stay the same while the mouse is held.
    public static class UI
    {
        public const int   TextSize = 20;
        public const float RowHeight = 36;
        public const float SliderHeight = 48;

        private const float Radius = 6;
        private const float Padding = 12;
        private const float TrackHeight = 6;

        // Greys out and locks interactive widgets drawn after it is set, Begin turns it back on.
        public static bool Enabled = true;

        private static int lastId = 0;
        private static int activeId = 0; // Widget the mouse went down on, it keeps the mouse until release

        public static void Begin()
        {
            lastId = 0;
            Enabled = true;
        }

        public static void End()
        {
            if (!Raylib.IsMouseButtonDown(MouseButton.Left))
                activeId = 0;
        }

        // Cuts a row off the top of area, for stacking widgets down a panel.
        public static Rectangle TakeTop(ref Rectangle area, float height, float gap = 8)
        {
            var row = new Rectangle(area.X, area.Y, area.Width, height);
            area.Y += height + gap;
            area.Height -= height + gap;
            return row;
        }

        // Returns the space left inside the panel.
        public static Rectangle Panel(Rectangle r, string? title = null)
        {
            DrawBox(r, Style.Panel, Style.PanelEdge);
            Rectangle inner = Style.Inflate(r, -Padding);
            if (title != null)
                Label(TakeTop(ref inner, TextSize), title, Style.TextDim);
            return inner;
        }

        public static void Label(Rectangle r, string text, Color? color = null)
        {
            DrawText(text, r.X, r.Y + (r.Height - TextSize) / 2, color ?? Style.Text);
        }

        public static bool Button(Rectangle r, string text)
        {
            bool clicked = Interact(r, out bool hovered, out bool held);

            Color fill = Style.PanelEdge;
            if (held) fill = Raylib.ColorBrightness(fill, -0.2f);
            else if (hovered) fill = Raylib.ColorBrightness(fill, 0.2f);

            DrawBox(r, Dim(fill), Dim(hovered ? Style.Highlight : Style.PanelEdge));
            Style.DrawTextCentered(text, Style.Center(r), TextSize, Dim(Style.Text));
            return clicked;
        }

        // Returns true on the frame the value changes.
        public static bool Toggle(Rectangle r, string text, ref bool value)
        {
            bool clicked = Interact(r, out bool hovered, out bool _);
            if (clicked) value = !value;

            float size = MathF.Min(r.Height, TextSize + 4);
            var box = new Rectangle(r.X, r.Y + (r.Height - size) / 2, size, size);
            DrawBox(box, Dim(Style.Background), Dim(hovered ? Style.Highlight : Style.PanelEdge));
            if (value)
                DrawBox(Style.Inflate(box, -5), Dim(Style.Highlight), Dim(Style.Highlight));

            DrawText(text, r.X + size + 8, r.Y + (r.Height - TextSize) / 2, Dim(Style.Text));
            return clicked;
        }

        // Returns true on the frame the value changes.
        public static bool Slider(Rectangle r, string text, ref double value, double min, double max, string format = "0.###")
        {
            double next = DrawSlider(r, text, value, min, max, value.ToString(format));
            bool changed = next != value;
            value = next;
            return changed;
        }

        public static bool Slider(Rectangle r, string text, ref int value, int min, int max, string format = "N0")
        {
            int next = (int)Math.Round(DrawSlider(r, text, value, min, max, value.ToString(format)));
            bool changed = next != value;
            value = next;
            return changed;
        }

        // Items that don't fit in r are not drawn. Returns true on the frame the selection changes.
        public static bool ListBox(Rectangle r, string[] items, ref int selected)
        {
            bool changed = false;
            for (int i = 0; i < items.Length && (i + 1) * RowHeight <= r.Height; i++)
            {
                var row = new Rectangle(r.X, r.Y + i * RowHeight, r.Width, RowHeight);
                if (Interact(row, out bool hovered, out bool _) && selected != i)
                {
                    selected = i;
                    changed = true;
                }

                if (i == selected)
                    DrawBox(row, Dim(Style.PanelEdge), Dim(Style.Highlight));
                else if (hovered)
                    DrawBox(row, Dim(Style.PanelEdge), Dim(Style.PanelEdge));

                DrawText(items[i], row.X + Padding, row.Y + (row.Height - TextSize) / 2, Dim(Style.Text));
            }
            return changed;
        }

        public static void ProgressBar(Rectangle r, double fraction, string? text = null)
        {
            DrawBox(r, Style.Background, Style.PanelEdge);
            float width = r.Width * (float)Math.Clamp(fraction, 0, 1);
            if (width >= 1)
                DrawBox(new Rectangle(r.X, r.Y, width, r.Height), Style.Highlight, Style.Highlight);
            if (text != null)
                DrawOutlinedText(text, Style.Center(r));
        }

        // Values are averaged into one point per pixel, so long noisy runs still read as a trend.
        public static void LineChart(Rectangle r, IReadOnlyList<double> values, string? title = null, Color? color = null)
        {
            Rectangle plot = Panel(r);
            Rectangle header = TakeTop(ref plot, TextSize);
            if (title != null)
                Label(header, title, Style.TextDim);

            if (values.Count == 0)
            {
                Style.DrawTextCentered("No data", Style.Center(plot), TextSize, Style.TextDim);
                return;
            }

            int count = Math.Min(values.Count, Math.Max(1, (int)plot.Width));
            double[] points = new double[count];
            double min = double.MaxValue;
            double max = double.MinValue;
            for (int i = 0; i < count; i++)
            {
                int start = (int)((long)i * values.Count / count);
                int end = (int)((long)(i + 1) * values.Count / count);
                for (int j = start; j < end; j++)
                    points[i] += values[j];
                points[i] /= end - start;
                min = Math.Min(min, points[i]);
                max = Math.Max(max, points[i]);
            }

            Vector2 PointAt(int i)
            {
                float x = count == 1 ? plot.X + plot.Width / 2 : plot.X + plot.Width * i / (count - 1);
                double height = max > min ? (points[i] - min) / (max - min) : 0.5;
                return new Vector2(x, plot.Y + plot.Height * (float)(1 - height));
            }

            Color line = color ?? Style.Highlight;
            if (count == 1)
                Raylib.DrawCircleV(PointAt(0), 3, line);
            for (int i = 1; i < count; i++)
                Raylib.DrawLineEx(PointAt(i - 1), PointAt(i), 2, line);

            string rangeText = max > min ? $"{min:0.###} to {max:0.###}" : min.ToString("0.###");
            DrawText(rangeText, header.X + header.Width - Raylib.MeasureText(rangeText, TextSize), header.Y, Style.TextDim);
        }

        // Returns the value after this frame's dragging.
        private static double DrawSlider(Rectangle r, string text, double value, double min, double max, string valueText)
        {
            Interact(r, out bool hovered, out bool held);

            float trackY = r.Y + r.Height - TrackHeight - 4;
            var track = new Rectangle(r.X, trackY, r.Width, TrackHeight);

            if (held)
            {
                double t = (Raylib.GetMousePosition().X - track.X) / track.Width;
                value = min + Math.Clamp(t, 0, 1) * (max - min);
            }

            float knobX = track.X + track.Width * (float)((value - min) / (max - min));
            Color accent = Dim(held || hovered ? Style.Highlight : Style.Text);

            DrawText(text, r.X, r.Y, Dim(Style.TextDim));
            DrawText(valueText, r.X + r.Width - Raylib.MeasureText(valueText, TextSize), r.Y, Dim(Style.Text));
            DrawBox(track, Dim(Style.Background), Dim(Style.PanelEdge));
            Raylib.DrawCircleV(new Vector2(knobX, trackY + TrackHeight / 2), 8, accent);
            return value;
        }

        // Returns true on the frame of a click, which is a press and release both inside r.
        private static bool Interact(Rectangle r, out bool hovered, out bool held)
        {
            int id = ++lastId;

            hovered = Enabled
                && Raylib.CheckCollisionPointRec(Raylib.GetMousePosition(), r)
                && (activeId == 0 || activeId == id);

            if (hovered && Raylib.IsMouseButtonPressed(MouseButton.Left))
                activeId = id;

            held = Enabled && activeId == id;
            return held && hovered && Raylib.IsMouseButtonReleased(MouseButton.Left);
        }

        private static Color Dim(Color c) => Enabled ? c : Raylib.Fade(c, 0.4f);

        // Corners keep the same radius whatever the size of the box.
        private static void DrawBox(Rectangle r, Color fill, Color edge)
        {
            float roundness = MathF.Min(1, 2 * Radius / MathF.Min(r.Width, r.Height));
            Raylib.DrawRectangleRounded(r, roundness, 6, fill);
            Raylib.DrawRectangleRoundedLinesEx(r, roundness, 6, 1.5f, edge);
        }

        private static void DrawText(string text, float x, float y, Color color)
        {
            Raylib.DrawText(text, (int)x, (int)y, TextSize, color);
        }

        // Stays readable over both the filled and empty parts of a bar.
        private static void DrawOutlinedText(string text, Vector2 center)
        {
            Style.DrawTextCentered(text, center + new Vector2(1, 1), TextSize, Style.Background);
            Style.DrawTextCentered(text, center, TextSize, Style.Text);
        }
    }
}
