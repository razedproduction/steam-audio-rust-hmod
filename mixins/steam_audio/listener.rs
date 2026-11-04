// Runs at the head of `Audio::set_listener` (each camera update, on the game thread): the support module takes the
// sound output over once per output Horizon opens; the rest of the time it only checks. The method then runs as
// before.
fn steam_audio_take_over(audio: &mut Audio) {
    steam_audio::take_over(audio);
}
