// mf/variations.h -- terry's transform presets.
//
// Generated from the service's variations.py by tools/gen_variations.py; do not
// edit by hand. Each preset is a target prompt plus the flow step to invert to,
// which is the only per-preset parameter -- everything else is the same euler/25
// setting for all of them.
#pragma once

#include <cstddef>

namespace ac {

struct Variation {
    const char* name;
    const char* prompt;
    double flowstep;
    int steps;
};

inline const Variation* variations(size_t& count) {
    static const Variation table[] = {
        {"accordion_folk",
         "Lively accordion music with a European folk feeling, perfect for a travel documentary about traditional culture and street performances in Paris",
         0.12, 25},
        {"banjo_bluegrass",
         "Authentic bluegrass banjo band performance with rich picking patterns, ideal for a heartfelt documentary about American rural life and traditional crafts",
         0.12, 25},
        {"piano_classical",
         "Expressive classical piano performance with dynamic range and emotional depth, ideal for a luxury brand commercial",
         0.12, 25},
        {"celtic",
         "Traditional Celtic arrangement with fiddle and flute, perfect for a documentary about Ireland's stunning landscapes and ancient traditions",
         0.12, 25},
        {"strings_quartet",
         "Elegant string quartet arrangement with rich harmonies and expressive dynamics, perfect for wedding ceremony music",
         0.12, 25},
        {"synth_retro",
         "1980s style synthesizer melody with warm analog pads and arpeggios, perfect for a nostalgic sci-fi movie soundtrack",
         0.12, 25},
        {"synth_modern",
         "Modern electronic production with crisp digital synthesizer arpeggios and vocoder effects, ideal for a tech product launch video",
         0.12, 25},
        {"synth_ambient",
         "Atmospheric synthesizer pads with reverb and delay, perfect for a meditation app or wellness commercial",
         0.12, 25},
        {"synth_edm",
         "High-energy EDM synth saw leads with sidechain compression, pitch bends, perfect for sports highlights or action sequences",
         0.12, 25},
        {"rock_band",
         "Full rock band arrangement with electric guitars, bass, and drums, perfect for an action movie trailer",
         0.12, 25},
        {"cinematic_epic",
         "Epic orchestral arrangement with modern hybrid elements, synthesizers, and percussion, perfect for movie trailers",
         0.12, 25},
        {"lofi_chill",
         "Lo-fi hip hop style with vinyl crackle, mellow piano, and tape saturation, perfect for study or focus playlists",
         0.12, 25},
        {"synth_bass",
         "Deep analog synthesizer bassline with modern production and subtle modulation, perfect for electronic music production",
         0.12, 25},
        {"retro_rpg",
         "16-bit era JRPG soundtrack with bright melodic synthesizers, orchestral elements, and adventurous themes, perfect for a fantasy video game battle scene or overworld exploration",
         0.12, 25},
        {"steel_drums",
         "Vibrant Caribbean steel drum ensemble with tropical percussion and uplifting melodies, perfect for a beach resort commercial or travel documentary",
         0.12, 25},
        {"chiptune",
         "8-bit video game soundtrack with arpeggiated melodies and classic NES-style square waves, perfect for a retro platformer or action game",
         0.12, 25},
        {"gamelan_fusion",
         "Indonesian gamelan ensemble with metallic percussion, gongs, and ethereal textures, perfect for a meditation app or spiritual documentary",
         0.12, 25},
        {"music_box",
         "Delicate music box melody with gentle bell tones and ethereal ambiance, perfect for a children's lullaby or magical fantasy scene",
         0.12, 25},
        {"trap_808",
         "Modern trap beat with booming 808 bass, crisp hi-hat rolls, and punchy snares, perfect for a contemporary hip hop music video with dramatic slow motion scenes",
         0.12, 25},
        {"lo_fi_drums",
         "Vinyl-processed lo-fi hip hop drums with warm tape saturation, subtle sidechain compression, and occasional vinyl crackle, ideal for relaxing study focus videos or late night coding sessions",
         0.12, 25},
        {"boom_bap",
         "Classic 90s boom bap hip hop drums with punchy kicks, crisp snares, and jazz sample chops, perfect for documentary footage of urban street scenes and skateboarding",
         0.12, 25},
        {"percussion_ensemble",
         "Rich percussive ensemble with djembe, congas, shakers, and tribal drums creating complex polyrhythms, perfect for nature documentaries about rainforests or ancient cultural rituals",
         0.12, 25},
        {"future_bass",
         "Energetic future bass with filtered supersaws, pitch-bending lead synths, heavy sidechain, and chopped vocal samples, perfect for extreme sports highlights or uplifting motivational content",
         0.12, 25},
        {"synthwave_retro",
         "80s retrofuturistic synthwave with gated reverb drums, analog arpeggios, neon-bright lead synths and driving bass, perfect for cyberpunk-themed technology showcases or retro gaming montages",
         0.12, 25},
        {"melodic_techno",
         "Hypnotic melodic techno with pulsing bass, atmospheric pads, and evolving synthesizer sequences with subtle filter modulation, ideal for timelapse footage of urban nightscapes or architectural showcases",
         0.12, 25},
        {"dubstep_wobble",
         "Heavy dubstep with aggressive wobble bass, metallic synthesizers, distorted drops, and tension-building risers, perfect for action sequence transitions or gaming highlight reels",
         0.12, 25},
        {"glitch_hop",
         "Glitch hop with stuttering sample slices, bit-crushed percussion, granular synthesis textures and digital artifacts, perfect for technology malfunction scenes or data visualization animations",
         0.12, 25},
        {"digital_disruption",
         "Heavily glitched soundscape with digital artifacts, buffer errors, granular time stretching, and corrupted audio samples, ideal for cybersecurity themes or digital distortion transitions in tech presentations",
         0.12, 25},
        {"circuit_bent",
         "Circuit-bent toy sounds with unpredictable pitch shifts, broken electronic tones, and hardware malfunction artifacts, perfect for creative coding demonstrations or innovative technology exhibitions",
         0.12, 25},
        {"orchestral_glitch",
         "Cinematic orchestral elements disrupted by digital glitches, granular textures, and temporal distortions, perfect for science fiction trailers or futuristic product reveals with contrasting classical and modern elements",
         0.12, 25},
        {"vapor_drums",
         "Vaporwave drum processing with extreme pitch and time manipulation, reverb-drenched samples, and retro commercial music elements, ideal for nostalgic internet culture documentaries or retrofuturistic art installations",
         0.12, 25},
        {"industrial_textures",
         "Harsh industrial soundscape with mechanical percussion, factory recordings, metallic impacts, and distorted synth drones, perfect for manufacturing process videos or dystopian urban environments",
         0.12, 25},
        {"jungle_breaks",
         "High-energy jungle drum breaks with choppy breakbeat samples, deep sub bass, and dub reggae influences, perfect for fast-paced urban chase scenes or extreme sports montages",
         0.12, 25},
    };
    count = sizeof(table) / sizeof(table[0]);
    return table;
}

// The preset by name, or null. Callers may still override the flow step and the
// prompt: terry's /transform takes custom_flowstep and custom_prompt alongside a
// preset name, and a custom prompt replaces the preset's rather than adding to it.
inline const Variation* find_variation(const char* name) {
    size_t count = 0;
    const Variation* table = variations(count);
    for (size_t i = 0; i < count; ++i) {
        const char* a = table[i].name;
        const char* b = name;
        while (*a && *a == *b) { ++a; ++b; }
        if (!*a && !*b) return &table[i];
    }
    return nullptr;
}

} // namespace ac
