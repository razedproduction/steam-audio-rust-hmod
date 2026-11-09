// Runs at the head of `native_hud` (the `hud` hook point: the HUD, not drawn under F1): the visual debug mode's labels
// over the sources and its statistics panel, in the first free corner. With every layer off it does nothing.
fn steam_audio_hud(game: &mut crate::game::Game<'_>, batch: &mut HudBatch, corners: &mut Corners) {
    use horizon_client_render::audio::steam_audio::{HudDraw, debug_hud};

    fn draw(game: &crate::game::Game<'_>, batch: &mut HudBatch, d: &HudDraw, at: [f32; 2]) {
        match d {
            HudDraw::Fill(r, argb) => {
                game.native_fill(batch, [r[0] + at[0], r[1] + at[1], r[2], r[3]], *argb);
            }
            HudDraw::Text(text, p, rgb) => {
                game.native_text(batch, text, [p[0] + at[0], p[1] + at[1]], *rgb, true);
            }
        }
    }

    let Some(hud) = debug_hud(&game.settings.language, batch.width, batch.height, &|text| {
        game.native_text_width(text)
    }) else {
        return;
    };
    for d in &hud.labels {
        draw(game, batch, d, [0.0, 0.0]);
    }
    if let Some((size, panel)) = &hud.panel
        && let Some(corner) = [
            Corner::TopLeft,
            Corner::BottomLeft,
            Corner::TopRight,
            Corner::BottomRight,
        ]
        .into_iter()
        .find(|&corner| game.native_corner_shown(corner))
    {
        let (x, y) = corners.place(corner, size[0], size[1]);
        for d in panel {
            draw(game, batch, d, [x, y]);
        }
    }
}
