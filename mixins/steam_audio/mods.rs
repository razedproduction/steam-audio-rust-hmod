// Runs at the head of `native_mods` (the `mods` hook point): lists the mod in Options > Mods, which gives it a
// settings page.
fn steam_audio_list(game: &mut crate::game::Game<'_>, mods: &mut Vec<NativeMod>) {
    let (id, title, description) =
        horizon_client_render::audio::steam_audio::mod_row(&game.settings.language);
    mods.push(NativeMod {
        id,
        title,
        description,
        keys: Vec::new(),
    });
}
