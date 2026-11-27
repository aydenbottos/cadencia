#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

struct AuroraUniforms {
    float time;
    float audioEnergy;
    float moodIndex;   // 0=Studio, 1=Warmup Saffron, 2=Input1 Velvet, 3=Input2 Midnight, 4=Output Jade/Gold
    float aspect;
};

struct OrbUniforms {
    float time;
    float energy;
    float state;       // 0=Idle, 1=Listening (mic), 2=Thinking (Terra), 3=Speaking (Phonon / Kokoro)
    float aspect;
    float4 bands0;
    float4 bands1;
};

struct StarPoint {
    float2 pos;
    float state;       // 0=New, 1=Learning, 2=Known, 3=Productive
    float recall;      // 0..1
    float isDue;       // 0 or 1
    float phase;
};

struct ConstellationUniforms {
    float time;
    float zoom;
    float2 pan;
    float aspect;
    float pad;
};

// Fullscreen triangle vertex shader
vertex VertexOut fullscreenVertex(uint vid [[vertex_id]]) {
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2( 3.0, -1.0),
        float2(-1.0,  3.0)
    };
    VertexOut out;
    out.position = float4(positions[vid], 0.0, 1.0);
    out.uv = positions[vid] * 0.5 + 0.5;
    return out;
}

static inline float hash21(float2 p) {
    p = fract(p * float2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

static inline float vnoise(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    float2 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash21(i + float2(0.0, 0.0)), hash21(i + float2(1.0, 0.0)), u.x),
        mix(hash21(i + float2(0.0, 1.0)), hash21(i + float2(1.0, 1.0)), u.x),
        u.y
    );
}

static inline float fbm(float2 p) {
    float v = 0.0;
    float a = 0.52;
    float2x2 rot = float2x2(0.80, -0.60, 0.60, 0.80);
    for (int i = 0; i < 4; ++i) {
        v += a * vnoise(p);
        p = rot * p * 2.05 + float2(1.7, 9.2);
        a *= 0.5;
    }
    return v;
}

// 1. Living Mediterranean Aurora & Silk Caustics Backdrop (120Hz ProMotion)
fragment float4 mediterraneanAuroraFragment(VertexOut in [[stage_in]],
                                            constant AuroraUniforms& u [[buffer(0)]]) {
    float2 uv = in.uv;
    float2 p = (uv - 0.5) * float2(u.aspect, 1.0);
    float t = u.time * 0.14;

    float2 q = float2(fbm(p * 1.4 + float2(t, t * 0.6)),
                      fbm(p * 1.4 + float2(-t * 0.5, t * 0.8) + 4.3));
    float f = fbm(p * 1.8 + 1.85 * q + float2(t * 0.35, -t * 0.25));

    // Mood-adaptive palettes
    float3 baseCol, glowA, glowB;
    int mood = int(round(u.moodIndex));
    if (mood == 1) {
        // Warm-up: Espresso & Saffron Dawn
        baseCol = float3(0.075, 0.055, 0.048);
        glowA   = float3(0.86, 0.42, 0.20);
        glowB   = float3(0.95, 0.74, 0.32);
    } else if (mood == 2) {
        // Input 1: Terracotta Nocturne Velvet
        baseCol = float3(0.068, 0.064, 0.082);
        glowA   = float3(0.76, 0.31, 0.21);
        glowB   = float3(0.90, 0.65, 0.36);
    } else if (mood == 3) {
        // Input 2 (Audio-First): Midnight Indigo & Bioluminescent Cobalt
        baseCol = float3(0.042, 0.058, 0.125);
        glowA   = float3(0.16, 0.45, 0.92);
        glowB   = float3(0.28, 0.78, 0.88);
    } else if (mood == 4) {
        // Output Speaking: Deep Obsidian, Mediterranean Jade & Warm Gold
        baseCol = float3(0.045, 0.078, 0.085);
        glowA   = float3(0.18, 0.72, 0.56);
        glowB   = float3(0.92, 0.68, 0.28);
    } else {
        // Home Studio: Obsidian, Terracotta & Cobalt Horizon
        baseCol = float3(0.055, 0.062, 0.095);
        glowA   = float3(0.82, 0.36, 0.24);
        glowB   = float3(0.24, 0.54, 0.92);
    }

    float caustic = pow(clamp(f, 0.0, 1.0), 1.85);
    float ribbon = smoothstep(0.32, 0.85, fbm(p * 2.5 - q * 1.4 + t));
    float energyBoost = 1.0 + u.audioEnergy * 0.55;

    float3 col = baseCol;
    col += glowA * caustic * 0.34 * energyBoost;
    col += glowB * ribbon * 0.22 * energyBoost;

    // Gentle vignette & ultra-fine film grain to eliminate 8-bit gradient banding
    float vig = 1.0 - dot(p * 0.78, p * 0.78);
    col *= clamp(vig, 0.48, 1.08);
    float grain = (hash21(uv * 913.7 + fract(u.time)) - 0.5) * 0.014;
    col += grain;

    return float4(clamp(col, 0.0, 1.0), 1.0);
}

// 2. Raymarched Iridescent Liquid-Glass Acoustic Resonance Orb
static inline float orbSDF(float3 p, float t, float energy, float state) {
    float r = 0.66 + energy * 0.085;
    float freq = (state > 0.5 && state < 1.5) ? 4.4 : ((state > 2.5) ? 3.2 : 2.2);
    float amp  = 0.025 + energy * 0.075;
    float disp = sin(p.x * freq + t * 2.4) * cos(p.y * freq - t * 1.9) * sin(p.z * freq + t * 1.5) * amp;
    disp += sin(length(p.xy) * 7.0 - t * 4.5) * energy * 0.028;
    return length(p) - r + disp;
}

static inline float3 orbNormal(float3 p, float t, float energy, float state) {
    float e = 0.0025;
    return normalize(float3(
        orbSDF(p + float3(e, 0, 0), t, energy, state) - orbSDF(p - float3(e, 0, 0), t, energy, state),
        orbSDF(p + float3(0, e, 0), t, energy, state) - orbSDF(p - float3(0, e, 0), t, energy, state),
        orbSDF(p + float3(0, 0, e), t, energy, state) - orbSDF(p - float3(0, 0, e), t, energy, state)
    ));
}

fragment float4 resonanceOrbFragment(VertexOut in [[stage_in]],
                                     constant OrbUniforms& u [[buffer(0)]]) {
    float2 uv = (in.uv - 0.5) * 2.0;
    uv.x *= u.aspect;

    float3 ro = float3(0.0, 0.0, 2.25);
    float3 rd = normalize(float3(uv, -1.65));

    // State-driven palette
    float3 primaryCol, secondaryCol;
    if (u.state > 0.5 && u.state < 1.5) {
        // Listening (Learner Microphone): Mediterranean Jade & Cobalt
        primaryCol   = float3(0.18, 0.85, 0.66);
        secondaryCol = float3(0.24, 0.58, 1.00);
    } else if (u.state >= 1.5 && u.state < 2.5) {
        // Thinking (GPT-6.1 Terra): Saffron Gold & Terracotta
        primaryCol   = float3(0.96, 0.72, 0.28);
        secondaryCol = float3(0.88, 0.36, 0.24);
    } else if (u.state >= 2.5) {
        // Speaking (Phonon-40m-es / Kokoro-82M): Solar Amber & Rose Coral
        primaryCol   = float3(0.98, 0.62, 0.28);
        secondaryCol = float3(0.90, 0.34, 0.42);
    } else {
        // Idle / Audio-first calm: Cobalt & Warm Pearl
        primaryCol   = float3(0.32, 0.64, 0.98);
        secondaryCol = float3(0.92, 0.76, 0.48);
    }

    // Soft outer volumetric acoustic halo
    float distCenter = length(uv);
    float haloRing = exp(-pow(abs(distCenter - (0.72 + u.energy * 0.14)) * 5.2, 2.0)) * (0.35 + u.energy * 0.75);
    float softAura = exp(-distCenter * distCenter * 2.2) * (0.28 + u.energy * 0.45);
    float3 bgGlow = mix(primaryCol, secondaryCol, 0.5 + 0.5 * sin(u.time + uv.x * 2.0)) * (softAura + haloRing);

    // Raymarch sphere
    float dist = 0.0;
    bool hit = false;
    float3 pos = ro;
    for (int i = 0; i < 48; ++i) {
        pos = ro + rd * dist;
        float d = orbSDF(pos, u.time, u.energy, u.state);
        if (d < 0.002) { hit = true; break; }
        if (dist > 4.2) break;
        dist += d * 0.85;
    }

    if (!hit) {
        float alpha = clamp(length(bgGlow) * 1.1, 0.0, 0.95);
        return float4(bgGlow, alpha);
    }

    float3 n = orbNormal(pos, u.time, u.energy, u.state);
    float3 v = -rd;
    float fresnel = pow(1.0 - max(dot(n, v), 0.0), 3.1);

    float3 light1 = normalize(float3(0.7, 0.85, 0.9));
    float3 light2 = normalize(float3(-0.8, -0.5, 0.6));
    float spec1 = pow(max(dot(reflect(-light1, n), v), 0.0), 42.0);
    float spec2 = pow(max(dot(reflect(-light2, n), v), 0.0), 18.0);

    // Iridescent thin-film phase shift inside glass core
    float swirl = fbm(n.xy * 2.6 + float2(u.time * 0.35, -u.time * 0.25));
    float3 coreColor = mix(primaryCol, secondaryCol, swirl);
    float3 refracted = mix(coreColor * 0.35, float3(0.98, 0.95, 0.90), fresnel * 0.75);

    float3 finalCol = refracted
        + coreColor * (0.45 + u.energy * 0.65) * (0.4 + 0.6 * max(dot(n, light1), 0.0))
        + float3(1.0, 0.97, 0.92) * spec1 * 0.95
        + secondaryCol * spec2 * 0.55
        + primaryCol * fresnel * (0.75 + u.energy * 0.55);

    return float4(clamp(finalCol, 0.0, 1.0), 0.98);
}

// 3. GPU-Instanced 4,500-Item Memory Constellation Vertex & Fragment Shaders
struct StarVertexOut {
    float4 position [[position]];
    float pointSize [[point_size]];
    float4 color;
    float pulse;
};

vertex StarVertexOut memoryStarVertex(uint vid [[vertex_id]],
                                      constant StarPoint* stars [[buffer(0)]],
                                      constant ConstellationUniforms& u [[buffer(1)]]) {
    StarPoint s = stars[vid];
    float2 p = (s.pos + u.pan) * u.zoom;
    p.x /= max(u.aspect, 0.5);

    float breathe = 0.5 + 0.5 * sin(u.time * 2.4 + s.phase);
    float size = 4.0;
    float4 col;

    if (s.state < 0.5) {
        // New: faint slate starlight
        col = float4(0.55, 0.60, 0.72, 0.24);
        size = 3.2;
    } else if (s.state < 1.5) {
        // Learning: warm terracotta ember
        col = float4(0.90, 0.42, 0.26, 0.75 + 0.20 * s.recall);
        size = 5.6;
    } else if (s.state < 2.5) {
        // Known: luminous bioluminescent cobalt
        col = float4(0.32, 0.70, 1.00, 0.88);
        size = 6.4;
    } else {
        // Productive: imperial saffron gold
        col = float4(0.98, 0.78, 0.34, 0.98);
        size = 8.0;
    }

    if (s.isDue > 0.5) {
        col = mix(col, float4(1.0, 0.82, 0.38, 1.0), 0.45 + 0.35 * breathe);
        size += 3.5 * breathe;
    }

    StarVertexOut out;
    out.position = float4(p, 0.0, 1.0);
    out.pointSize = size * clamp(u.zoom, 0.75, 2.4);
    out.color = col;
    out.pulse = s.isDue > 0.5 ? breathe : 0.0;
    return out;
}

fragment float4 memoryStarFragment(StarVertexOut in [[stage_in]],
                                   float2 pointCoord [[point_coord]]) {
    float d = length(pointCoord - float2(0.5)) * 2.0;
    if (d > 1.0) discard_fragment();
    float core = exp(-d * d * 4.5);
    float halo = exp(-d * d * 1.6) * (0.45 + 0.35 * in.pulse);
    float alpha = clamp((core + halo) * in.color.a, 0.0, 1.0);
    float3 rgb = in.color.rgb + float3(0.35) * core;
    return float4(clamp(rgb, 0.0, 1.0), alpha);
}
